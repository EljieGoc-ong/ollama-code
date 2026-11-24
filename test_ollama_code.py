"""Self-check for ollama-code. Needs no model: the agent loop runs against a fake Ollama server.

Run with `python test_ollama_code.py` (or `pytest`).
"""
import _thread
import io
import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from ollama_code import agent, attach, cli, client, mcp, terminal, tools

APP = Path(__file__).with_name("run.py")  # the no-install launcher


def test_tools():
    with tempfile.TemporaryDirectory() as d:
        crlf = Path(d) / "a.txt"
        crlf.write_bytes(b"one\r\ntwo\r\ntwo\r\n")
        assert "occurs 2 times" in tools.edit_file(str(crlf), "two", "2")
        assert tools.edit_file(str(crlf), "one\ntwo", "1\n2").startswith("Edited")  # \n edit on a CRLF file
        assert crlf.read_bytes() == b"1\r\n2\r\ntwo\r\n"
        assert "not found" in tools.edit_file(str(crlf), "zzz", "y")
        assert tools.edit_file(str(crlf), "two", "2", replace_all="true").startswith("Edited")

        tools.write_file(str(Path(d) / "sub" / "b.py"), "print('hi')\nx = 1\n")
        assert tools.list_files("**/*.py", d) == "sub/b.py"
        assert tools.grep("PRINT", d, glob="*.py", ignore_case=True) == "sub/b.py:1: print('hi')"
        assert "     2\tx = 1" in tools.read_file(str(Path(d) / "sub" / "b.py"))
        assert "exit code 0" in tools.run_command("echo hello") and "hello" in tools.run_command("echo hello")
        assert tools.execute_tool("nope", {}).startswith("Error: unknown tool")
        assert tools.execute_tool("read_file", {"path": str(Path(d) / "missing.txt")}).startswith("Error:")


def test_repair_history():
    msgs = [{"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "grep"}},
                                                                 {"function": {"name": "read_file"}}]},
            {"role": "tool", "tool_name": "grep", "content": "ok"}]
    agent.repair_history(msgs)
    assert msgs[-1] == {"role": "tool", "tool_name": "read_file", "content": "Interrupted by the user."}


class FakeOllama(BaseHTTPRequestHandler):
    """Turn 1: the model asks to edit app.py. Turn 2: it answers in text."""
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        msg = {"role": "assistant", "content": ""}
        if len(FakeOllama.requests) == 1:
            call = {"function": {"name": "edit_file",
                                 "arguments": {"path": "app.py", "old_string": "Hello", "new_string": "Goodbye"}}}
            chunks = [{"message": {**msg, "thinking": "I should edit app.py"}, "done": False},
                      {"message": {**msg, "tool_calls": [call]}, "done": False},
                      {"message": msg, "done": True, "prompt_eval_count": 50, "eval_count": 9,
                       "eval_duration": 1_000_000_000}]
        else:
            chunks = [{"message": {**msg, "content": "Done, "}, "done": False},
                      {"message": {**msg, "content": "greeting changed."}, "done": True,
                       "prompt_eval_count": 70, "eval_count": 4, "eval_duration": 1_000_000_000}]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for c in chunks:
            self.wfile.write((json.dumps(c) + "\n").encode())

    def log_message(self, *args):
        pass


def _run_agent(extra_args, workdir=None, home=None, prompt="change the greeting"):
    """Run the CLI once against the fake server. Sessions go to a temp home, never the real one."""
    FakeOllama.requests = []
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(workdir or tmp)
        if not (d / "app.py").exists():
            (d / "app.py").write_text("print('Hello')\n")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OLLAMA_CODE_HOME": str(home or Path(tmp) / "home"),
               "CLAUDE_CONFIG_DIR": str(Path(tmp) / "claude")}  # keep the developer's own skills/agents out
        try:
            proc = subprocess.run([sys.executable, str(APP), "--host", f"127.0.0.1:{server.server_port}",
                                   "-p", prompt, *extra_args],
                                  cwd=d, capture_output=True, text=True, encoding="utf-8",
                                  stdin=subprocess.DEVNULL, env=env, timeout=60)
        finally:
            server.shutdown()
        return proc, (d / "app.py").read_text(), FakeOllama.requests


def test_agent_loop_with_permission():
    proc, app, reqs = _run_agent(["--yolo"])
    assert proc.returncode == 0, proc.stderr
    assert app == "print('Goodbye')\n"
    shown = proc.stdout
    for expected in ("✻ Thinking", "I should edit app.py",          # reasoning is streamed by default
                     "● Update(app.py)", "- Hello", "+ Goodbye",     # the call and its diff
                     "⎿  Edited", "● Done, greeting changed.",      # the result and the answer
                     "↑ 120 in · ↓ 13 out · 6.5 tok/s · context 74"):  # the usage footer
        assert expected in shown, f"missing {expected!r} in:\n{shown}"
    first, second = reqs
    assert first["model"] == "glm-4.7-flash" and "options" not in first  # don't force a model reload
    assert {t["function"]["name"] for t in first["tools"]} == set(tools.TOOLS)
    assert second["messages"][-2]["tool_calls"][0]["function"]["name"] == "edit_file"
    assert second["messages"][-1]["role"] == "tool" and second["messages"][-1]["content"].startswith("Edited")


def test_agent_loop_denied_without_terminal():
    proc, app, reqs = _run_agent([])
    assert proc.returncode == 0, proc.stderr
    assert app == "print('Hello')\n"  # nothing changed
    assert "denied" in reqs[1]["messages"][-1]["content"]


def test_hide_thinking():
    proc, _app, _reqs = _run_agent(["--yolo", "--hide-thinking"])
    assert "I should edit app.py" not in proc.stdout and "Done, greeting changed." in proc.stdout


def test_sessions_save_continue_and_resume():
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as home:
        proc, _app, _reqs = _run_agent(["--yolo"], workdir=work, home=home)
        assert proc.returncode == 0, proc.stderr
        files = list((Path(home) / "sessions").glob("*.json"))
        assert len(files) == 1
        saved = json.loads(files[0].read_text(encoding="utf-8"))
        assert saved["title"] == "change the greeting" and saved["usage"]["output"] == 13
        assert saved["messages"][0] == {"role": "user", "content": "change the greeting"}  # no system prompt stored

        # -c picks the conversation back up: the model sees the earlier turn before the new prompt.
        proc, _app, reqs = _run_agent(["--yolo", "-c"], workdir=work, home=home, prompt="and now?")
        assert proc.returncode == 0, proc.stderr
        assert "Resumed session" in proc.stdout
        sent = [m.get("content") for m in reqs[0]["messages"]]
        assert sent[1] == "change the greeting" and sent[-1] == "and now?"
        assert len(list((Path(home) / "sessions").glob("*.json"))) == 1  # same session, updated in place

        # -r ID works from a different folder, and says where the session came from.
        with tempfile.TemporaryDirectory() as elsewhere:
            proc, _app, reqs = _run_agent(["--yolo", "-r", saved["id"][:15]], workdir=elsewhere, home=home)
        assert "Resumed session" in proc.stdout and "this session was started in" in proc.stdout
        assert reqs[0]["messages"][1]["content"] == "change the greeting"


class StallingOllama(BaseHTTPRequestHandler):
    """Sends one chunk, then goes silent like a model writing a tool call, and notices when the client hangs up."""
    hung_up = threading.Event()

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"message": {"role": "assistant", "thinking": "hmm"}, "done": false}\n')
        self.wfile.flush()
        deadline = time.time() + 20
        while time.time() < deadline:
            readable, _, _ = select.select([self.connection], [], [], 0.1)
            if readable:
                try:
                    if self.connection.recv(1) == b"":
                        break
                except OSError:
                    break
        else:
            return
        StallingOllama.hung_up.set()

    def log_message(self, *args):
        pass


def test_ctrl_c_works_while_ollama_is_silent():
    StallingOllama.hung_up.clear()
    server = HTTPServer(("127.0.0.1", 0), StallingOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Timer(1.0, _thread.interrupt_main).start()  # simulates pressing Ctrl+C
    started = time.monotonic()
    try:
        for _chunk in client.stream_chat(f"http://127.0.0.1:{server.server_port}", {"model": "m"}):
            pass
        raise AssertionError("stream ended without being interrupted")
    except KeyboardInterrupt:
        pass
    assert time.monotonic() - started < 3, "Ctrl+C took too long to take effect"
    assert StallingOllama.hung_up.wait(5), "connection was not closed, so Ollama would keep generating"
    server.shutdown()


def test_status_line():
    real, buf = sys.stdout, io.StringIO()
    sys.stdout = buf
    try:
        s = terminal.Screen(live=False)
        s.set_status("Thinking")
        s.last_write = time.monotonic() - 1
        s.tick(0)
        assert "Thinking… (" in buf.getvalue() and s.drawn
        s.write("hello\n")
        assert buf.getvalue().endswith("\r\033[Khello\n")  # status line erased before printing
        s.write("mid-line")
        s.last_write = time.monotonic() - 3
        s.tick(1)
        assert "mid-line\n\r\033[K" in buf.getvalue()  # long pause mid-line: status goes on a new line
        s.set_status(None)
        before = buf.getvalue()
        s.last_write = 0
        s.tick(2)
        assert buf.getvalue() == before  # hidden status draws nothing
    finally:
        sys.stdout = real


def test_pinned_header_tracks_rows():
    real, buf = sys.stdout, io.StringIO()
    sys.stdout = buf
    try:
        s = terminal.Screen(live=False)
        s.live = True  # allow pinning without starting the animation thread
        s.pin("Thinking")
        width = terminal.shutil.get_terminal_size((80, 20)).columns
        s.write("one\ntwo\n" + "x" * (width + 1))  # two newlines plus one soft wrap
        assert s.rows_below == 3
        s.tick(0)
        assert "\0337\033[4A\r\033[K" in buf.getvalue()  # jumped up past 3 rows + the header itself
        s.unpin("done")
        assert s.pinned is None and buf.getvalue().endswith("done\0338")
    finally:
        sys.stdout = real


def test_attachments():
    with tempfile.TemporaryDirectory() as d:
        spaced = Path(d) / "my notes.txt"
        spaced.write_text("hello")
        (Path(d) / "pic.png").write_bytes(b"\x89PNG")
        escaped = spaced.as_posix().replace(" ", "\\ ")  # how macOS/Linux terminals paste a dragged path
        prompt = f'look at "{spaced}", {escaped}, @{Path(d) / "pic.png"} and @nope.py'
        found = attach.attachments(prompt)
        assert [(p.name, k) for p, k, _ in found] == [("my notes.txt", "text"), ("pic.png", "image")], found
        msg = attach.build_message("hi", found)
        assert "hello\n</file>" in msg["content"] and len(msg["images"]) == 1
    assert attach.attachments("fix README.md please") == []  # bare relative words are not attachments
    assert attach.match_files(["src/app.py", "lib/apple.txt", "readme.md"], "app") == ["src/app.py", "lib/apple.txt"]


def test_suggestions():
    s = cli.Suggester()
    s.files = ["src/app.py", "docs/guide.md"]
    hint = s("/re", 3)
    assert hint.submit and [t for t, _ in hint.items] == ["/resume"]
    assert s("/model x", 8).items == []  # typing arguments: no more suggestions
    hint = s("explain @ap", 11)
    assert hint.start == 8 and hint.items == [("@src/app.py", "")] and not hint.submit


def test_line_editor_keys():
    from contextlib import nullcontext
    from ollama_code import lineedit
    saved = lineedit.AVAILABLE, lineedit._windows_keys, lineedit._unix_keys, lineedit._raw_mode, sys.stdout
    s = cli.Suggester()
    s.files = ["src/app.py"]

    def typed(*keys):
        script = iter(keys)
        lineedit._windows_keys = lineedit._unix_keys = lambda: script
        return lineedit.read_line("> ", s)

    try:
        lineedit.AVAILABLE, lineedit._raw_mode, sys.stdout = True, nullcontext, io.StringIO()
        assert typed("/", "r", "e", "\r") == "/resume"                 # Enter runs the highlighted command
        assert typed("/", "\t", "x", "\r") == "/help x"                # Tab only completes
        assert typed("s", "e", "e", " ", "@", "a", "\r", "\r") == "see @src/app.py "  # Enter picks the file
        assert typed("a", "c", "left", "b", "\r") == "abc"
        assert typed("up", "\r") == "abc"                              # history
        for key, error in (("\x03", KeyboardInterrupt), ("\x04", EOFError)):
            try:
                typed(key)
                raise AssertionError(f"{key!r} should raise {error.__name__}")
            except error:
                pass
    finally:
        lineedit.AVAILABLE, lineedit._windows_keys, lineedit._unix_keys, lineedit._raw_mode, sys.stdout = saved


def test_claude_skills_and_agents():
    from ollama_code import extensions
    saved_home, saved_cwd = extensions.CLAUDE_HOME, os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        home = Path(d) / "claude"
        (home / "skills" / "tidy").mkdir(parents=True)
        (home / "skills" / "tidy" / "SKILL.md").write_text(
            "---\nname: tidy\ndescription: |\n  Clean up code\n  carefully.\nmetadata:\n  origin: x\n---\nTidy $ARGUMENTS now.\n")
        (home / "agents").mkdir()
        (home / "agents" / "reviewer.md").write_text(
            "---\nname: reviewer\ndescription: Reviews code\ntools: [\"Read\", \"Grep\", \"mcp__docs__*\"]\n---\nYou review.\n")
        (Path(d) / "proj" / ".claude" / "agents").mkdir(parents=True)  # project agents win over user ones
        (Path(d) / "proj" / ".claude" / "agents" / "reviewer.md").write_text("---\nname: reviewer\n---\nProject one.\n")
        try:
            extensions.CLAUDE_HOME = home
            os.chdir(Path(d) / "proj")
            assert extensions.load() == (1, 1)
            assert extensions.skills["tidy"]["description"] == "Clean up code carefully."
            assert "Tidy the imports now." in extensions.skill_text("tidy", "the imports")
            assert extensions.agent_prompt("reviewer")[0] == "Project one."
            assert "Did you mean: tidy" in extensions.skill_text("tid")
            assert "- tidy: Clean up code carefully." in extensions.catalog()
            assert {s["function"]["name"] for s in extensions.schemas()} == {"use_skill", "run_agent"}
            spec = '["Read", "Grep", "mcp__docs__*"]'
            assert extensions.allowed_tools(spec, ["read_file", "run_command", "grep", "mcp__docs__search"]) == \
                ["read_file", "grep", "mcp__docs__search"]
            assert extensions.allowed_tools("", ["a", "b"]) == ["a", "b"]
            assert cli.skill_request("/tidy imports").startswith("Follow the 'tidy' skill")
            assert cli.skill_request("/help") is None and cli.skill_request("/nope") is None
        finally:
            os.chdir(saved_cwd)
            extensions.CLAUDE_HOME = saved_home
            extensions.skills.clear()
            extensions.agents.clear()


def test_run_agent_subloop():
    from ollama_code import extensions
    seen, saved = [], (agent.model_reply, dict(extensions.agents))
    replies = iter([
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "write_file",
                                                                        "arguments": {"path": "x", "content": "y"}}}]},
        {"role": "assistant", "content": "Report: looks fine."},
    ])

    def fake_reply(state):
        seen.append(([s["function"]["name"] for s in agent.tool_schemas(state)], list(state.messages)))
        return next(replies), {}

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "reviewer.md"
        path.write_text("---\nname: reviewer\ntools: Read, Grep\n---\nYou review code.\n")
        extensions.agents["reviewer"] = {"name": "reviewer", "description": "", "path": path, "tools": "Read, Grep"}
        try:
            agent.model_reply = fake_reply
            parent = agent.State("m", "http://x", 0, True, True, yolo=True)
            before = list(parent.messages)
            report = agent.run_agent(parent, "reviewer", "review app.py")
        finally:
            agent.model_reply = saved[0]
            extensions.agents.clear()
            extensions.agents.update(saved[1])
    assert report == "Report: looks fine."
    assert seen[0][0] == ["read_file", "grep"]                     # tools cut down to the agent's list
    assert "You review code." in seen[0][1][0]["content"] and seen[0][1][1]["content"] == "review app.py"
    assert "isn't allowed to use write_file" in seen[1][1][-1]["content"]  # even with --yolo
    assert parent.messages == before                               # the sub-agent's chat stays separate
    assert "Error: no agent named 'nope'" in agent.run_agent(parent, "nope", "x")


def test_schedule_parsing_and_timing():
    from ollama_code import schedule
    now = 1_000_000.0
    assert schedule.parse_duration("1h30m") == 5400 and schedule.parse_duration("soon") is None
    assert schedule.parse_schedule("in 10m run tests", now) == (now + 600, None, "run tests")
    assert schedule.parse_schedule("every 2s ping", now)[1] == schedule.MIN_INTERVAL  # clamped
    due, every, prompt = schedule.parse_schedule("at 9:15pm sum up", now)
    assert every is None and prompt == "sum up" and 0 < due - now <= 86400
    assert time.localtime(due).tm_hour == 21 and time.localtime(due).tm_min == 15
    for bad in ("in 10m", "at 25:00 x", "tomorrow x"):
        try:
            schedule.parse_schedule(bad, now)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert schedule.parse_loop("5m check build", now) == (now, 300, "check build")
    assert schedule.parse_loop("check build", now) == (now, schedule.DEFAULT_LOOP, "check build")
    saved = list(schedule.jobs)
    schedule.jobs.clear()
    try:
        loop = schedule.add(now, 300, "check", kind="loop")
        once = schedule.add(now + 60, None, "later")
        assert schedule.take_due(now) == [loop] and loop.due == now + 300   # loop moves to its next slot
        assert schedule.take_due(now + 1000) == [once, loop]                 # missed slots run once, not 3x
        assert once not in schedule.jobs and loop.due == now + 1200
        assert schedule.cancel("all", kind="loop") == 1 and not schedule.jobs
    finally:
        schedule.jobs[:] = saved


def _scripted(replies):
    """A stand-in for agent.model_reply that answers from a list, recording each request's tools."""
    script, seen = iter(replies), []

    def fake(state):
        seen.append([s["function"]["name"] for s in agent.tool_schemas(state)])
        return next(script), {}
    return fake, seen


def _call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_background_agent():
    from ollama_code import extensions, tasks
    saved = agent.model_reply, dict(extensions.agents)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "scout.md"
        path.write_text("---\nname: scout\n---\nYou scout.\n")
        extensions.agents["scout"] = {"name": "scout", "description": "", "path": path, "tools": ""}
        try:
            agent.model_reply, _ = _scripted([_call("run_command", command="echo hi"),
                                              {"role": "assistant", "content": "Found 3 things."}])
            parent = agent.State("m", "http://x", 0, True, True, yolo=False)
            started = agent.call_tool(parent, "run_agent", {"agent": "scout", "task": "look", "background": True})
            assert started.startswith("Started background task #")
            job = list(tasks.tasks.values())[-1]
            for _ in range(100):
                if job.status != "running":
                    break
                time.sleep(0.05)
            assert job.status == "done" and job.report == "Found 3 things."
            assert tasks.has_finished() and tasks.take_finished() == [job] and not tasks.has_finished()
            assert "Found 3 things." in tasks.notice(job) and len(parent.messages) == 1  # parent untouched
            child = agent.make_subagent(parent, "scout", "x")
            child.background = threading.Event()
            assert "needs the user's approval" in agent.background_tool(child, "run_command", {"command": "x"})
            assert agent.background_tool(child, "list_files", {"pattern": "*.nothing"}) == "No files matched."
        finally:
            agent.model_reply = saved[0]
            extensions.agents.clear()
            extensions.agents.update(saved[1])


def test_goal_runs_until_complete():
    from ollama_code import automation
    saved = agent.model_reply
    try:
        agent.model_reply, seen = _scripted([
            {"role": "assistant", "content": "Did part one."},                    # round 1: not done yet
            _call("goal_complete", summary="All tests pass."),                    # round 2: done
            {"role": "assistant", "content": "Finished."}])
        state = agent.State("m", "http://x", 0, True, True, yolo=True)
        rounds = []

        def run_prompt(s, text):
            rounds.append(text)
            agent.run_turn(s, text)
            return True

        automation.goal_command(state, "make the tests pass", run_prompt)
    finally:
        agent.model_reply = saved
    assert len(rounds) == 2 and rounds[0].startswith("New goal") and "Round 2" in rounds[1]
    assert "goal_complete" in seen[0] and state.goal is None
    assert state.goal_result == ("All tests pass.", False)


def test_prompt_wakes_for_background_work():
    from contextlib import nullcontext
    from ollama_code import lineedit
    saved = lineedit.AVAILABLE, lineedit._key_ready, lineedit._raw_mode, lineedit._windows_keys, lineedit._unix_keys
    real_stdout = sys.stdout
    try:
        lineedit.AVAILABLE, lineedit._raw_mode, sys.stdout = True, nullcontext, io.StringIO()
        lineedit._key_ready = lambda timeout: False   # nobody is typing
        lineedit._windows_keys = lineedit._unix_keys = lambda: iter(())
        try:
            lineedit.read_line("> ", lambda b, p: lineedit.NONE, wake=lambda: True)
            raise AssertionError("should have woken up")
        except lineedit.Wake:
            pass
    finally:
        sys.stdout = real_stdout
        (lineedit.AVAILABLE, lineedit._key_ready, lineedit._raw_mode,
         lineedit._windows_keys, lineedit._unix_keys) = saved


FAKE_MCP_SERVER = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method, result = msg["method"], {}
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "Echo text back",
                             "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
                            {"name": "peek", "annotations": {"readOnlyHint": True}}]}
    elif method == "tools/call":
        print("not json, a stray log line", flush=True)
        result = {"content": [{"type": "text", "text": "echo: " + msg["params"]["arguments"]["text"]}]}
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
'''


def test_mcp_stdio_server():
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "server.py"
        script.write_text(FAKE_MCP_SERVER)
        server = mcp.Server("fake", {"command": sys.executable, "args": [str(script)]})
        mcp.servers["fake"] = server
        try:
            server.start()
            assert server.status == "ready", server.status
            assert [f["function"]["name"] for f in mcp.schemas()] == ["mcp__fake__echo", "mcp__fake__peek"]
            assert mcp.call("mcp__fake__echo", {"text": "hi"}) == "echo: hi"
            assert mcp.needs_approval("mcp__fake__echo") and not mcp.needs_approval("mcp__fake__peek")
            server.stop()
            server.proc.wait(10)
            assert mcp.call("mcp__fake__echo", {"text": "hi"}).startswith("Error: MCP server fake")
        finally:
            server.stop()
            mcp.servers.pop("fake", None)


def test_unreachable_server():
    with tempfile.TemporaryDirectory() as home:
        proc = subprocess.run([sys.executable, str(APP), "--host", "127.0.0.1:9", "-p", "hi"],
                              capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
                              env={**os.environ, "OLLAMA_CODE_HOME": home})
    assert proc.returncode == 1 and "Can't reach Ollama" in proc.stdout


def test_live_token_counter():
    s = terminal.Screen(live=False)
    s.new_turn()
    for _ in range(5):
        s.note_token()
    s.step_first, s.step_last = 0.0, 2.0  # 4 gaps between 5 tokens over 2 seconds
    assert "↓ 5 tokens · 2.0 tok/s · Ctrl+C to interrupt" in s.progress()


def test_terminal_detection_never_crashes():
    assert terminal._is_msys_pty(sys.stdout) in (True, False)
    assert terminal._is_msys_pty(io.StringIO()) is False  # no real file handle


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
