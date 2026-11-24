"""Talking to Ollama: streaming /api/chat, checking and preloading the model."""
import http.client
import json
import queue
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request



class OllamaError(Exception):
    pass


def _hang_up(conn, sock):
    """Close the connection, waking a reader blocked on it. Ollama then stops generating.

    Takes the socket separately because http.client drops conn.sock once a response owns it.
    """
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    conn.close()


def stream_chat(host, payload):
    """POST to /api/chat and yield each streamed JSON chunk.

    The HTTP read runs on a helper thread while this one waits in short slices, so Ctrl+C works
    even while Ollama is silent (a blocked socket read can't be interrupted on Windows). Leaving
    early, for any reason, hangs up so Ollama doesn't keep generating for nobody.
    """
    url = urllib.parse.urlsplit(host)
    conn_class = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    conn = conn_class(url.hostname or "localhost", url.port)
    try:
        conn.request("POST", url.path.rstrip("/") + "/api/chat", json.dumps(payload).encode("utf-8"),
                     {"Content-Type": "application/json"})
    except OSError as e:
        conn.close()
        raise OllamaError(f"Can't reach Ollama at {host} ({e}). Is Ollama running?") from None

    sock = conn.sock
    lines = queue.Queue()

    def read():
        try:
            resp = conn.getresponse()
            if resp.status != 200:
                body = resp.read().decode("utf-8", "replace")
                try:
                    body = json.loads(body).get("error", body)
                except ValueError:
                    pass
                raise OllamaError(f"Ollama returned HTTP {resp.status}: {body}")
            for line in resp:
                lines.put(line)
            lines.put(None)
        except Exception as e:  # handed over to the main thread
            lines.put(e)

    threading.Thread(target=read, daemon=True).start()
    try:
        while True:
            try:
                item = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            if item.strip():
                chunk = json.loads(item)
                if "error" in chunk:
                    raise OllamaError(chunk["error"])
                yield chunk
    finally:
        _hang_up(conn, sock)


def loaded_model(state):
    """Ollama's /api/ps entry for our model if it is in memory, {} if not, None if Ollama is unreachable."""
    try:
        with urllib.request.urlopen(state.host.rstrip("/") + "/api/ps", timeout=5) as r:
            models = json.load(r).get("models", [])
    except (OSError, ValueError):
        return None
    wanted = state.model if ":" in state.model else state.model + ":latest"
    return next((m for m in models if m.get("name") in (state.model, wanted)), {})


def loaded_context(state):
    """Context length of the model as Ollama actually loaded it, or 0 if unknown."""
    return (loaded_model(state) or {}).get("context_length", 0)


def preload(state):
    """Ask Ollama to load the model in the background, so it's ready by the time the user has typed."""
    body = {"model": state.model}
    if state.ctx:
        body["options"] = {"num_ctx": state.ctx}
    req = urllib.request.Request(state.host.rstrip("/") + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})

    def load():
        try:
            urllib.request.urlopen(req, timeout=900).close()
        except OSError:
            pass  # the first real request will report any problem

    threading.Thread(target=load, daemon=True).start()
