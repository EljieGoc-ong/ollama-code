"""Model Context Protocol: run the stdio MCP servers from mcp.json and offer their tools to the model.

Config uses the same shape as Claude Code's .mcp.json, read from ~/.ollama-code/mcp.json and then
./.mcp.json (the project file wins on name clashes):

    {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                           "env": {"SOME_TOKEN": "..."}}}}
"""
import atexit
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

PROTOCOL_VERSION = "2025-06-18"
START_TIMEOUT = 60   # seconds for a server to start and list its tools (npx may download first)
CALL_TIMEOUT = 300   # seconds for one tool call
HOME = Path(os.environ.get("OLLAMA_CODE_HOME") or Path.home() / ".ollama-code")


class McpError(Exception):
    pass


def tool_name(server, tool):
    """Ollama function names: letters, digits, _ and - only."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp__{server}__{tool}")


class Server:
    def __init__(self, name, config):
        self.name, self.config = name, config
        self.status, self.tools, self.proc = "starting", {}, None  # tools: model-facing name -> MCP tool
        self.replies, self.lock, self.next_id = queue.Queue(), threading.Lock(), 0

    def start(self):
        try:
            if "command" not in self.config:
                raise McpError("only stdio servers (with a \"command\") are supported")
            command = shutil.which(self.config["command"]) or self.config["command"]  # finds npx.cmd on Windows
            self.proc = subprocess.Popen(
                [command, *map(str, self.config.get("args", []))], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env={**os.environ, **self.config.get("env", {})},
                cwd=self.config.get("cwd") or None)
            threading.Thread(target=self._read, daemon=True).start()
            self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                        "clientInfo": {"name": "ollama-code", "version": "0.3.0"}}, START_TIMEOUT)
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            listed = self.request("tools/list", {}, START_TIMEOUT)  # ponytail: first page only, no cursor paging
            self.tools = {tool_name(self.name, t["name"]): t for t in listed.get("tools", [])}
            self.status = "ready"
        except (OSError, McpError, KeyError, TypeError, ValueError) as e:
            self.status = f"failed: {e}"
            self.stop()

    def _read(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue  # servers sometimes log to stdout; ignore non-JSON lines
            if isinstance(msg, dict) and "method" in msg and "id" in msg:  # a request to us; we offer nothing
                self._send({"jsonrpc": "2.0", "id": msg["id"],
                            "error": {"code": -32601, "message": "not supported by ollama-code"}})
            elif isinstance(msg, dict) and "id" in msg:
                self.replies.put(msg)
        self.replies.put(None)  # stdout closed: the server exited

    def _send(self, msg):
        self.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def request(self, method, params, timeout):
        with self.lock:  # ponytail: one request at a time per server, fine for a single agent
            self.next_id += 1
            self._send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
            deadline = time.monotonic() + timeout
            while True:
                try:
                    msg = self.replies.get(timeout=0.2)  # short slices: Esc / Ctrl+C get through on Windows
                except queue.Empty:
                    if time.monotonic() > deadline:
                        raise McpError(f"{self.name} didn't answer {method} within {timeout}s") from None
                    continue
                if msg is None:
                    raise McpError(f"{self.name} exited")
                if msg.get("id") == self.next_id:
                    break
        if "error" in msg:
            raise McpError((msg["error"] or {}).get("message", "unknown error"))
        return msg.get("result") or {}

    def call(self, name, args):
        result = self.request("tools/call", {"name": self.tools[name]["name"], "arguments": args}, CALL_TIMEOUT)
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "resource":
                parts.append((item.get("resource") or {}).get("text", "[resource]"))
            else:
                parts.append(f"[{item.get('type', 'unknown')} content omitted]")
        if not parts and "structuredContent" in result:
            parts.append(json.dumps(result["structuredContent"]))
        text = "\n".join(parts) or "(no output)"
        return f"Error: {text}" if result.get("isError") else text

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


servers = {}


def load_config():
    merged = {}
    for path in (HOME / "mcp.json", Path.cwd() / ".mcp.json"):
        if path.is_file():
            try:
                merged.update(json.loads(path.read_text(encoding="utf-8")).get("mcpServers", {}))
            except (ValueError, AttributeError) as e:
                servers[path.name] = _broken(path.name, f"failed: can't read {path}: {e}")
    return merged


def _broken(name, status):
    s = Server(name, {})
    s.status = status
    return s


def start_all():
    """Start every configured server in the background; their tools appear as each one becomes ready."""
    for name, config in load_config().items():
        if config.get("disabled"):
            continue
        servers[name] = Server(name, config)
        threading.Thread(target=servers[name].start, daemon=True).start()
    atexit.register(lambda: [s.stop() for s in servers.values()])
    return len(servers)


def wait_until_started(timeout=START_TIMEOUT):
    """For one-shot -p runs, which have no typing time to hide server start-up behind."""
    deadline = time.monotonic() + timeout
    while any(s.status == "starting" for s in servers.values()) and time.monotonic() < deadline:
        time.sleep(0.1)


def _ready():
    return [s for s in servers.values() if s.status == "ready"]


def owner(name):
    return next((s for s in _ready() if name in s.tools), None)


def schemas():
    return [{"type": "function", "function": {
        "name": name, "description": f"[{s.name} MCP] " + (t.get("description") or t["name"]),
        "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}}}
        for s in _ready() for name, t in s.tools.items()]


def needs_approval(name):
    """Ask before every MCP tool unless the server marks it read-only."""
    s = owner(name)
    return not (s and (s.tools[name].get("annotations") or {}).get("readOnlyHint"))


def label(name, args):
    s = owner(name)
    short = json.dumps(args, ensure_ascii=False)
    return f"{s.name} · {s.tools[name]['name']}({short[:80] + '…' if len(short) > 80 else short})"


def call(name, args):
    s = owner(name)
    try:
        return s.call(name, args)
    except (McpError, OSError) as e:  # OSError: the server died and its pipe broke
        return f"Error: MCP server {s.name}: {e}"
