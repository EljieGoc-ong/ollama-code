"""The interactive prompt, slash commands and command-line options."""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from .terminal import (ACCENT, COLOR, TTY, bold, cyan, dim, gradient, green, out, red, rgb,
                       screen, shimmer, visible_len, yellow)
from . import extensions, lineedit, mcp
from .attach import attachments, match_files, project_files
from .lineedit import Suggestions
from .tools import (_clip, run_command)
from .client import OllamaError, loaded_model, preload
from .agent import (State, compact, footer, repair_history, run_turn, show_usage)
from .sessions import _same_dir, list_sessions, load_session, resume, save_session

VERSION = "0.3.0"
DEFAULT_MODEL = "glm-4.7-flash"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_CTX = 0  # 0 = use the context Ollama already has loaded; forcing a size reloads the model


COMMANDS = [  # (command, arguments, what it does): drives /help and the live suggestions
    ("/help", "", "show this help"),
    ("/clear", "", "start a fresh conversation"),
    ("/compact", "", "summarize the conversation to free up context"),
    ("/usage", "", "tokens used and speed for this session"),
    ("/resume", "[n|id]", "list saved sessions, or resume one (sessions save after every turn)"),
    ("/model", "[name]", "show or switch the model"),
    ("/mcp", "", "list MCP servers and their tools"),
    ("/skills", "", "list the Claude Code skills found (run one with /<skill> [request])"),
    ("/agents", "", "list the Claude Code agents the model can delegate to"),
    ("/think", "", "toggle showing the model's reasoning (shown by default)"),
    ("/yolo", "", "toggle asking before edits and commands"),
    ("/exit", "", "quit (also Ctrl+D, or Ctrl+Z then Enter on Windows)"),
]
HELP = "Commands:\n" + "\n".join(f"  {(c + ' ' + a).strip():<17}{d}" for c, a, d in COMMANDS) + """
  !<command>       run a shell command yourself; its output is added to the conversation
  @path            attach a file (type @ for suggestions, or drag a file into the terminal)
End a line with \\ to keep typing on the next line. Ctrl+C stops the current response."""


class Suggester:
    """Live suggestions for the prompt: /commands at the start of the line, @files anywhere."""

    def __init__(self):
        self.files = None  # listed on the first @ of each prompt, then reused while typing

    def __call__(self, buf, pos):
        before = buf[:pos]
        if before.startswith("/") and " " not in before:
            typed = before.lower()
            return Suggestions(0, [(c, d) for c, _a, d in COMMANDS if c.startswith(typed)]
                               + [("/" + n, s["description"]) for n, s in extensions.skills.items()
                                  if ("/" + n.lower()).startswith(typed) or ("/" + n.split(":")[-1].lower()).startswith(typed)],
                               submit=True)
        word_start = max(before.rfind(" "), before.rfind("\t")) + 1
        word = before[word_start:]
        if not word.startswith("@") or word.startswith('@"'):
            return lineedit.NONE
        if self.files is None:
            self.files = project_files()
        return Suggestions(word_start, [("@" + f, "") for f in match_files(self.files, word[1:])])


def skill_request(text):
    """`/skill-name request` -> the prompt that runs it, or None if no such skill (or it's a built-in)."""
    cmd, _, arg = text[1:].partition(" ")
    if any(cmd.lower() == c[1:] for c, _a, _d in COMMANDS):
        return None
    name = cmd if cmd in extensions.skills else next(  # plugin skills also answer to their short name
        (n for n in extensions.skills if n.split(":")[-1] == cmd), None)
    if not name:
        return None
    return (f"Follow the '{name}' skill below for this request: {arg.strip() or '(no extra details given)'}\n\n"
            + extensions.skill_text(name, arg.strip()))


def handle_command(state, text):
    """Run a /command. Returns False when the app should exit."""
    cmd, _, arg = text.partition(" ")
    cmd, arg = cmd.lower(), arg.strip()
    if cmd in ("/exit", "/quit"):
        return False
    if cmd == "/help":
        out(HELP)
    elif cmd == "/clear":
        state.start_session()
        out(dim("Started a new conversation. The previous one is saved; /resume brings it back."))
    elif cmd == "/compact":
        guarded(state, lambda: compact(state))
    elif cmd in ("/usage", "/cost", "/stats"):
        show_usage(state)
    elif cmd in ("/resume", "/sessions"):
        resume(state, arg)
    elif cmd == "/model":
        if arg:
            state.model = arg
        out(dim(f"Model: {state.model}"))
    elif cmd == "/think":
        state.show_thinking = not state.show_thinking
        out(dim(f"Showing reasoning: {'on' if state.show_thinking else 'off'}"))
    elif cmd == "/yolo":
        state.yolo = not state.yolo
        out(dim(f"Ask before edits and commands: {'off' if state.yolo else 'on'}"))
    elif cmd == "/mcp":
        show_mcp()
    elif cmd in ("/skills", "/agents"):
        table = extensions.skills if cmd == "/skills" else extensions.agents
        if not table:
            out(dim(f"None found in .claude/{cmd[1:]}, {extensions.CLAUDE_HOME / cmd[1:]} or enabled plugins."))
        for name, item in table.items():
            out(f"  {bold(name)}  " + dim(item["description"][:100]))
    else:
        out(yellow(f"Unknown command {cmd}. Type /help for the list."))
    return True


def show_mcp():
    if not mcp.servers:
        out(dim(f"No MCP servers. Add them to .mcp.json here or {mcp.HOME / 'mcp.json'}:"))
        out(dim('  {"mcpServers": {"name": {"command": "npx", "args": ["-y", "some-mcp-server"]}}}'))
        return
    for s in mcp.servers.values():
        mark = green("●") if s.status == "ready" else yellow("◌") if s.status == "starting" else red("✗")
        out(f"{mark} {bold(s.name)} " + dim(s.status + (f" · {len(s.tools)} tools" if s.tools else "")))
        for t in s.tools.values():
            out(dim(f"    {t['name']}  {(t.get('description') or '').splitlines()[0][:70] if t.get('description') else ''}"))


def with_files(prompt):
    """Find files the prompt mentions (@path or a dragged-in path) and say which got attached."""
    try:
        files = attachments(prompt)
    except OSError as e:
        out(yellow(f"  couldn't read an attached file: {e}"))
        return []
    for path, kind, payload in files:
        size = "image" if kind == "image" else f"{payload.count(chr(10)) + 1:,} lines"
        out(dim(f"  ⎿  Attached {path.name} ({size})"))
    return files


def read_prompt():
    lines, suggest = [], Suggester()
    while True:
        line = screen.ask(rgb("❯ ", ACCENT[0], "1;") if not lines else dim("┊ "),
                          reader=lambda prompt: lineedit.read_line(prompt, suggest))
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def run_bang(state, command):
    screen.new_turn()
    screen.set_status("Running")
    try:
        result = run_command(command)
    finally:
        screen.set_status(None)
    out(dim(result))
    state.messages.append({"role": "user", "content": f"I ran this command myself:\n$ {command}\n{_clip(result)}"})
    save_session(state)


def guarded(state, action):
    """Run one model-backed action, turning Ctrl+C and errors into messages instead of a crash.
    The session is saved afterwards either way, so even an interrupted turn can be resumed."""
    ok = False
    try:
        action()
        ok = True
    except KeyboardInterrupt:
        out(yellow("\n[interrupted]"))
    except OllamaError as e:
        out(red(f"\nError: {e}"))
    except (OSError, ValueError) as e:  # dropped connection, malformed stream
        out(red(f"\nError talking to Ollama: {type(e).__name__}: {e}"))
    finally:
        screen.set_status(None)
    if not ok:
        repair_history(state.messages)
    try:
        save_session(state)
    except OSError as e:
        out(yellow(f"(couldn't save the session: {e})"))
    return ok


def animate_title(text):
    """Type the title out with a light sweeping across it. Plain print when not on a terminal."""
    if TTY and COLOR:
        for i in range(1, len(text) + 1):
            sys.stdout.write("\r" + shimmer(text[:i], i, base=ACCENT[0]))
            sys.stdout.flush()
            time.sleep(0.018)
        for frame in range(len(text), len(text) + 14, 2):
            sys.stdout.write("\r" + shimmer(text, frame, base=ACCENT[0]))
            sys.stdout.flush()
            time.sleep(0.02)
        sys.stdout.write("\r")
    out(gradient(text, extra="1;"))


def box(rows):
    """Rounded box around (label, value) rows; values may contain color codes."""
    room = shutil.get_terminal_size((80, 20)).columns - 4
    lines = [f"{dim(label.ljust(6))} {value}" for label, value in rows]
    width = min(room, max(visible_len(line) for line in lines))
    edge = lambda s: rgb(s, ACCENT[1])  # noqa: E731
    out(edge("╭" + "─" * (width + 2) + "╮"))
    for line in lines:
        out(edge("│ ") + line + " " * max(0, width - visible_len(line)) + edge(" │"))
    out(edge("╰" + "─" * (width + 2) + "╯"))


def banner(state):
    animate_title(f"✻ ollama-code v{VERSION}")
    entry = loaded_model(state)
    if entry is None:
        status = red("unreachable, start Ollama then send a message")
    elif entry:
        status = green("● loaded") + dim(f" · context {entry.get('context_length', 0):,}")
    else:
        status = yellow("◌ loading in the background")
        preload(state)
    cwd = str(Path.cwd())
    room = shutil.get_terminal_size((80, 20)).columns - 12
    if len(cwd) > room:  # keep the box one line per row on narrow terminals
        cwd = "…" + cwd[-(room - 1):]
    box([("model", bold(state.model) + "  " + status),
         ("host", dim(state.host)),
         ("cwd", cwd),
         ("mode", yellow("yolo, tools run without asking") if state.yolo else "asks before edits and commands")]
        + ([("mcp", f"{len(mcp.servers)} server{'s' * (len(mcp.servers) != 1)} starting · /mcp for status")]
           if mcp.servers else [])
        + ([("claude", f"{len(extensions.skills)} skills · {len(extensions.agents)} agents · /skills /agents")]
           if extensions.skills or extensions.agents else []))
    out(dim("  / commands · @ attach a file · !cmd runs shell · end a line with \\ for more"))
    out()


def repl(state):
    banner(state)
    ctrl_c_once = False
    while True:
        try:
            text = read_prompt()
            ctrl_c_once = False
        except EOFError:
            out()
            return
        except KeyboardInterrupt:
            if ctrl_c_once:
                out()
                return
            ctrl_c_once = True
            out(dim("\n(press Ctrl+C again or type /exit to quit)"))
            continue
        if not text:
            continue
        if text.startswith("/") and skill_request(text):
            out(dim(f"  ⎿  Running skill {text.split()[0][1:]}"))
            text = skill_request(text)
        elif text.startswith("/"):
            if handle_command(state, text) is False:
                return
            continue
        if text.startswith("!"):
            run_bang(state, text[1:].strip())
            continue
        files = with_files(text)
        if guarded(state, lambda: run_turn(state, text, files)):
            footer(state)
        out()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ollama-code", description="A Claude-Code-style coding agent powered by Ollama.")
    ap.add_argument("-m", "--model", default=os.environ.get("OLLAMA_CODE_MODEL", DEFAULT_MODEL),
                    help=f"Ollama model to use (default {DEFAULT_MODEL}, or $OLLAMA_CODE_MODEL)")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_CODE_HOST", DEFAULT_HOST),
                    help=f"Ollama server URL (default {DEFAULT_HOST}, or $OLLAMA_CODE_HOST)")
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("OLLAMA_CODE_CTX", DEFAULT_CTX)),
                    help="context window (num_ctx) to request from Ollama. Default 0 uses whatever Ollama has "
                         "loaded; setting a different size makes Ollama reload the model")
    ap.add_argument("--no-think", action="store_true", help="turn off the model's reasoning step (faster)")
    ap.add_argument("--hide-thinking", action="store_true", help="don't print the model's reasoning as it streams")
    ap.add_argument("--yolo", action="store_true", help="run edits and commands without asking")
    ap.add_argument("--no-claude", action="store_true",
                    help="don't load Claude Code skills and agents from .claude folders and plugins")
    ap.add_argument("-p", "--print", dest="prompt", metavar="PROMPT",
                    help="run a single prompt non-interactively, print the result and exit")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true",
                    help="continue the most recent session from this folder")
    ap.add_argument("-r", "--resume", nargs="?", const="", metavar="ID",
                    help="resume a saved session by id, or pick one from a list")
    ap.add_argument("--version", action="version", version=f"ollama-code {VERSION}")
    args = ap.parse_args(argv)

    host = args.host if "://" in args.host else "http://" + args.host
    if not args.no_claude:
        extensions.load()  # before State: the system prompt lists them
    state = State(args.model, host, args.ctx, not args.no_think, not args.hide_thinking, args.yolo)
    mcp.start_all()
    if args.cont:
        here = [s for s in list_sessions() if _same_dir(s.get("cwd"))]
        if here:
            load_session(state, here[0])
        else:
            out(dim("No earlier session in this folder, so starting a new one."))
    elif args.resume is not None:
        resume(state, args.resume)
    if args.prompt is not None:
        mcp.wait_until_started()
        files = with_files(args.prompt)
        if not guarded(state, lambda: run_turn(state, args.prompt, files)):
            return 1
        footer(state)
        return 0
    repl(state)
    return 0

