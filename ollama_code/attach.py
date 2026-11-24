"""Files in a prompt: @path mentions and paths dragged into the terminal get attached to the message."""
import base64
import os
import re
from pathlib import Path

from .tools import _walk

MAX_FILE_CHARS = 100_000
MAX_LISTED_FILES = 20_000  # stop walking huge trees; @ suggestions only need a decent sample
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# @"path with spaces", @path, "quoted path", 'quoted path', or a bare token
TOKEN = re.compile(r'@"([^"]+)"|@(\S+)|"([^"]+)"|\'([^\']+)\'|((?:\\ |\S)+)')


def _absolute(text):
    return Path(text).is_absolute() or text.startswith("~")


def _candidates(prompt):
    """Paths worth checking: every @mention, plus absolute paths (what drag-and-drop pastes)."""
    for m in TOKEN.finditer(prompt):
        mention = m.group(1) or m.group(2)
        if mention:
            yield mention.rstrip(".,;:!?)")
            continue
        text = m.group(3) or m.group(4) or (m.group(5) or "").replace("\\ ", " ")  # macOS/Linux escape spaces
        if text.startswith("file://"):
            text = text[7:]
        if _absolute(text):
            yield text


def attachments(prompt):
    """Files the prompt refers to that exist: [(path, kind, payload)], kind is 'text' or 'image'."""
    found, seen = [], set()
    for raw in _candidates(prompt):
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else Path.cwd() / path
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        data = path.read_bytes()
        if path.suffix.lower() in IMAGE_TYPES:
            found.append((path, "image", base64.b64encode(data).decode()))
        elif b"\0" not in data[:8000]:  # skip binaries
            found.append((path, "text", data.decode("utf-8", errors="replace")))
    return found


def build_message(prompt, files):
    """The user message with text files inlined and images passed the way Ollama expects."""
    message = {"role": "user", "content": prompt}
    for path, kind, payload in files:
        if kind == "image":
            message.setdefault("images", []).append(payload)
            continue
        body = payload if len(payload) <= MAX_FILE_CHARS else \
            payload[:MAX_FILE_CHARS] + f"\n... [truncated {len(payload) - MAX_FILE_CHARS} characters]"
        message["content"] += f'\n\n<file path="{path}">\n{body}\n</file>'
    return message


def project_files():
    files = []
    for path in _walk(Path.cwd()):
        files.append(path.relative_to(Path.cwd()).as_posix())
        if len(files) >= MAX_LISTED_FILES:
            break
    return files


def match_files(files, query, limit=8):
    """Prefix matches on the file name first, then anywhere in the path. Case-insensitive."""
    q = query.lower()
    by_name = [f for f in files if os.path.basename(f).lower().startswith(q)]
    rest = [f for f in files if q in f.lower() and f not in by_name]
    return (by_name + rest)[:limit]

