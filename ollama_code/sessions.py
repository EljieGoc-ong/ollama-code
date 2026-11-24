"""Saving conversations to disk and resuming them."""
import datetime
import json
import os
from pathlib import Path

from . import lineedit
from .terminal import (INTERACTIVE, bold, dim, green, out, screen, yellow)
from .agent import (describe, new_usage, system_prompt)
from .tools import parse_args


SESSIONS_DIR = Path(os.environ.get("OLLAMA_CODE_HOME") or Path.home() / ".ollama-code") / "sessions"


def _same_dir(path):
    return path and os.path.normcase(os.path.abspath(path)) == os.path.normcase(str(Path.cwd()))


def save_session(state):
    """Write the conversation to disk. Called after every turn, so nothing is lost on exit or a crash."""
    prompts = [m["content"] for m in state.messages if m["role"] == "user"]
    if not prompts:
        return
    usage = {k: v for k, v in state.usage.items() if k != "started"}
    data = {"id": state.session_id, "cwd": str(Path.cwd()), "model": state.model,
            "title": (prompts[0].strip().splitlines() or [""])[0][:80], "created": state.created,
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "usage": {**usage, "context": state.tokens},
            "inputs": state.inputs[-MAX_INPUTS:],  # what was typed, /commands and !commands too: the ↑ history
            "messages": state.messages[1:]}  # the system prompt is rebuilt fresh on resume
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{state.session_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic, so Ctrl+C mid-save can't leave a half-written session


def list_sessions():
    """All saved sessions, newest first."""
    sessions = []
    for p in SESSIONS_DIR.glob("*.json"):  # ponytail: parses every file; add an index if this gets slow
        try:
            sessions.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(sessions, key=lambda s: s.get("updated", ""), reverse=True)


def load_session(state, data):
    state.session_id, state.created = data["id"], data.get("created", "")
    state.messages = [{"role": "system", "content": system_prompt()}] + data.get("messages", [])
    saved = data.get("usage", {})
    state.usage = {**new_usage(), **{k: v for k, v in saved.items() if k != "context"}}
    state.tokens = saved.get("context", 0)
    state.inputs = list(data.get("inputs", []))
    lineedit.history[:] = state.inputs  # ↑ brings back what was typed in this session
    prompts = sum(1 for m in data.get("messages", []) if m.get("role") == "user")
    out(dim(f"Resumed session {data['id']} · {prompts} prompts · last active {data.get('updated', '?')}"))
    if not _same_dir(data.get("cwd")):
        out(yellow(f"Note: this session was started in {data.get('cwd')}"))
    for line in transcript(data.get("messages", []))[-REPLAY_LINES:]:
        out(line)


REPLAY_LINES = 14
MAX_INPUTS = 500


def transcript(messages):
    """One line per event, for the replay on resume: prompts, your !commands, the model's tool calls
    (commands it ran, files it edited) and its answers."""
    lines = []
    for m in messages:
        text = (m.get("content") or "").strip()
        if m.get("role") == "user" and text.startswith(BANG_PREFIX):
            lines.append(bold("! ") + text[len(BANG_PREFIX):].lstrip("$ ").splitlines()[0][:150])
        elif m.get("role") == "user" and text:
            lines.append(bold("> ") + text.splitlines()[0][:150])
        elif m.get("role") == "assistant":
            if text:
                lines.append(dim(f"● {text.splitlines()[0][:150]}"))
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                lines.append(green("● ") + dim(describe(fn.get("name", ""), parse_args(fn.get("arguments")))[:150]))
    return lines


BANG_PREFIX = "I ran this command myself:\n"


def describe_session(i, s):
    here = "" if _same_dir(s.get("cwd")) else dim(f"  [{s.get('cwd')}]")
    when = s.get("updated", "?").replace("T", " ")[:16]
    return f"  {i:>2}. {dim(when)}  {s.get('title') or '(untitled)'}{here}"


def resume(state, arg=""):
    """Resume a saved session: by list number, by id (or the start of one), or by picking from a list."""
    everything = list_sessions()
    here = [s for s in everything if _same_dir(s.get("cwd"))]
    shown = (here or everything)[:15]
    if not everything:
        out(dim("No saved sessions yet."))
        return False
    if not arg:
        out(dim("Sessions from this folder:" if here else "No sessions from this folder yet. All sessions:"))
        for i, s in enumerate(shown, 1):
            out(describe_session(i, s))
        if not INTERACTIVE:
            return False
        arg = screen.ask(yellow("Resume which one? (number, Enter to cancel): ")).strip()
        if not arg:
            return False
    if arg.isdigit() and 1 <= int(arg) <= len(shown):
        chosen = shown[int(arg) - 1]
    else:
        chosen = next((s for s in everything if s.get("id", "").startswith(arg)), None)
    if chosen is None:
        out(yellow(f"No session matches {arg!r}. Use /resume to list them."))
        return False
    load_session(state, chosen)
    return True
