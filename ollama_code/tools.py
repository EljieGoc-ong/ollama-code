"""The tools the model can call: read, write, edit, list, search files and run commands."""
import fnmatch
import json
import os
import re
import time
from pathlib import Path

from . import shells, web

MAX_TOOL_OUTPUT = 30_000  # chars of tool output sent back to the model
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
             ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build"}


def _resolve(path):
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else Path.cwd() / p


def _clip(text, limit=MAX_TOOL_OUTPUT):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} more characters]"


def _truthy(value):
    return value is True or str(value).strip().lower() in ("true", "1", "yes")


def _walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def read_file(path, offset=1, limit=2000):
    lines = _resolve(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(int(offset), 1) - 1
    chunk = lines[start:start + int(limit)]
    if not chunk:
        return "(empty file)" if not lines else f"(offset is past the end; file has {len(lines)} lines)"
    body = "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(chunk, start + 1))
    remaining = len(lines) - start - len(chunk)
    return body + (f"\n... ({remaining} more lines; read again with a larger offset)" if remaining > 0 else "")


def write_file(path, content):
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return f"Wrote {len(content.splitlines())} lines to {p}"


def edit_file(path, old_string, new_string, replace_all=False):
    p = _resolve(path)
    with open(p, encoding="utf-8", newline="") as f:
        text = f.read()
    if old_string not in text and "\r\n" in text and "\r\n" not in old_string:
        # The model writes \n; this file uses Windows line endings. Match and keep them.
        old_string, new_string = old_string.replace("\n", "\r\n"), new_string.replace("\n", "\r\n")
    count = text.count(old_string) if old_string else 0
    if count == 0:
        return "Error: old_string was not found. Read the file again and copy the text exactly, including indentation."
    replace_all = _truthy(replace_all)
    if count > 1 and not replace_all:
        return (f"Error: old_string occurs {count} times. Include more surrounding lines to make it unique, "
                "or set replace_all to true.")
    text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return f"Edited {p} ({count if replace_all else 1} replacement(s))"


def list_files(pattern="**/*", path="."):
    root = _resolve(path)
    short = pattern[3:] if pattern.startswith("**/") else pattern  # so "**/*.py" also matches top-level files
    hits = []
    for f in _walk(root):
        rel = f.relative_to(root).as_posix()
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, short) or fnmatch.fnmatch(f.name, pattern):
            hits.append(rel)
            if len(hits) >= 500:
                hits.append("... (stopped at 500 results; use a narrower pattern)")
                break
    return "\n".join(hits) or "No files matched."


def grep(pattern, path=".", glob=None, ignore_case=False):
    rx = re.compile(pattern, re.IGNORECASE if _truthy(ignore_case) else 0)
    root = _resolve(path)
    files = [root] if root.is_file() else _walk(root)
    hits = []
    for f in files:
        if glob and not fnmatch.fnmatch(f.name, glob):
            continue
        try:
            if f.stat().st_size > 2_000_000:
                continue
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable
        rel = f.name if root.is_file() else f.relative_to(root).as_posix()
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:300]}")
                if len(hits) >= 200:
                    return "\n".join(hits) + "\n... (stopped at 200 matches; narrow the search)"
    return "\n".join(hits) or "No matches."


def wait_for(shell, seconds):
    """Wait for a command in short slices so Ctrl+C is noticed on every platform."""
    deadline = time.monotonic() + float(seconds)
    while not shell.done.wait(0.2):
        if time.monotonic() > deadline:
            return False
    return True


def run_command(command, timeout=None, background=False):
    """Run a command; if it's still going after `timeout` seconds (or background is set) it keeps
    running in the background and its result is reported when it ends."""
    timeout = shells.FOREGROUND_WAIT if timeout is None else timeout
    shell = shells.start(command)
    if not _truthy(background):
        try:
            finished = wait_for(shell, timeout)
        except KeyboardInterrupt:
            shell.stop()
            raise
        if finished:
            return shell.result()
    shell.backgrounded = True
    so_far = shell.text(tail=2000)
    why = "" if _truthy(background) else f" (still going after {timeout}s)"
    return (f"Running in the background as shell #{shell.id}{why}. Its result arrives as a message when it "
            "ends; meanwhile carry on, check it with shell_output or end it with stop_shell."
            + (f"\nOutput so far:\n{so_far}" if so_far else ""))


def shell_output(shell, tail=4000):
    s = shells.find(shell)
    if not s:
        return f"Error: no shell #{shell}. Running ones: {', '.join(str(i) for i in shells.shells) or 'none'}"
    status = "still running" if s.running else f"finished with exit code {s.code}"
    return f"shell #{s.id} {status}: {s.command}\n{s.text(tail=int(tail))}".rstrip()


def stop_shell(shell):
    s = shells.find(shell)
    if not s:
        return f"Error: no shell #{shell}."
    if not s.running:
        return f"shell #{s.id} had already finished with exit code {s.code}."
    s.delivered = True  # the model asked for this; no need to report the end
    s.stop()
    return f"Stopped shell #{s.id}.\n{s.text(tail=2000)}".rstrip()


def _p(kind, description):
    return {"type": kind, "description": description}


# name -> (function, description, parameters, required, changes_state)
TOOLS = {
    "read_file": (read_file, "Read a text file. Returns lines prefixed with line numbers.", {
        "path": _p("string", "File path, absolute or relative to the working directory"),
        "offset": _p("integer", "1-based line to start from (default 1)"),
        "limit": _p("integer", "Maximum number of lines to return (default 2000)"),
    }, ["path"], False),
    "write_file": (write_file, "Create a file or overwrite it entirely. Parent folders are created.", {
        "path": _p("string", "File path"),
        "content": _p("string", "The complete file content"),
    }, ["path", "content"], True),
    "edit_file": (edit_file, "Replace an exact string in a file. old_string must match exactly, "
                             "including whitespace, and be unique unless replace_all is true.", {
        "path": _p("string", "File path"),
        "old_string": _p("string", "Exact text to replace"),
        "new_string": _p("string", "Replacement text"),
        "replace_all": _p("boolean", "Replace every occurrence (default false)"),
    }, ["path", "old_string", "new_string"], True),
    "list_files": (list_files, "List files matching a glob pattern such as '**/*.py' or 'src/*'. "
                               "Skips .git, node_modules, virtualenvs and caches.", {
        "pattern": _p("string", "Glob pattern (default '**/*')"),
        "path": _p("string", "Folder to search (default: working directory)"),
    }, [], False),
    "grep": (grep, "Search file contents with a regular expression. Returns file:line: text.", {
        "pattern": _p("string", "Python regular expression"),
        "path": _p("string", "File or folder to search (default: working directory)"),
        "glob": _p("string", "Only search files whose name matches this glob, e.g. '*.js'"),
        "ignore_case": _p("boolean", "Case-insensitive search (default false)"),
    }, ["pattern"], False),
    "run_command": (run_command, "Run a shell command in the working directory and return its exit code "
                                 "and output. Not interactive: stdin is closed. Commands that take longer than "
                                 "`timeout` seconds keep running in the background and report back when done; "
                                 "set background for servers, watchers and long builds.", {
        "command": _p("string", "The command line to run"),
        "timeout": _p("integer", f"Seconds to wait before moving it to the background "
                                 f"(default {shells.FOREGROUND_WAIT})"),
        "background": _p("boolean", "Start it in the background right away (default false)"),
    }, ["command"], True),
    "web_search": (web.web_search, "Search the web. Returns titles, URLs and snippets.", {
        "query": _p("string", "What to search for"),
        "count": _p("integer", "How many results (default 8)"),
    }, ["query"], False),
    "web_fetch": (web.web_fetch, "Fetch a web page (or any URL) and return its readable text and its links. "
                                 "JavaScript doesn't run, so script-built pages may come back thin.", {
        "url": _p("string", "The URL to read"),
        "max_chars": _p("integer", "Most characters of page text to return (default 20000)"),
    }, ["url"], True),
    "shell_output": (shell_output, "Show the latest output and status of a background command.", {
        "shell": _p("integer", "Shell number from run_command"),
        "tail": _p("integer", "How many characters of the most recent output to return (default 4000)"),
    }, ["shell"], False),
    "stop_shell": (stop_shell, "Stop a background command (and anything it started).", {
        "shell": _p("integer", "Shell number from run_command"),
    }, ["shell"], False),
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": name, "description": desc,
                                      "parameters": {"type": "object", "properties": params, "required": req}}}
    for name, (_fn, desc, params, req, _changes) in TOOLS.items()
]


def execute_tool(name, args):
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOLS)}"
    try:
        return _clip(str(TOOLS[name][0](**args)))
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"
    except Exception as e:  # report every failure to the model so it can recover
        return f"Error: {type(e).__name__}: {e}"


def parse_args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}
