"""The prompt's line editor: live suggestions (/commands, @files), arrow keys and history.

Standard library only: msvcrt on Windows, termios on Linux/macOS. Falls back to input() where
neither works (pipes, Git Bash's mintty), so suggestions are a bonus, never a requirement.
"""
import codecs
import contextlib
import os
import shutil
import sys
import time

from .terminal import ACCENT, IS_WINDOWS, TTY, dim, readline_safe, rgb, visible_len

MAX_SUGGESTIONS = 8
history = []

try:
    if IS_WINDOWS:
        import msvcrt
    else:
        import select
        import termios
    AVAILABLE = TTY and sys.stdin.isatty()
except (ImportError, AttributeError, ValueError):
    AVAILABLE = False


class Suggestions:
    """What to offer for the text before the cursor: replace buf[start:pos] with one of `items`.
    `items` are (text, description); `submit` means Enter runs the pick instead of just inserting it."""

    def __init__(self, start=0, items=(), submit=False):
        self.start, self.items, self.submit = start, list(items)[:MAX_SUGGESTIONS], submit


NONE = Suggestions()

_WIN_KEYS = {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end", "S": "delete"}
_ESC_KEYS = {"A": "up", "B": "down", "D": "left", "C": "right", "H": "home", "F": "end",
             "1~": "home", "7~": "home", "4~": "end", "8~": "end", "3~": "delete"}


def _windows_keys():
    while True:
        ch = msvcrt.getwch()
        yield _WIN_KEYS.get(msvcrt.getwch(), "") if ch in ("\x00", "\xe0") else ch


def _unix_keys():
    fd, decoder = sys.stdin.fileno(), codecs.getincrementaldecoder("utf-8")(errors="replace")

    def getch():
        while True:
            text = decoder.decode(os.read(fd, 1))
            if text:
                return text

    while True:
        ch = getch()
        if ch != "\x1b":
            yield ch
        elif not select.select([fd], [], [], 0.03)[0]:
            yield "esc"
        elif getch() in "[O":
            seq = getch()
            while not (seq[-1].isalpha() or seq[-1] == "~"):
                seq += getch()
            yield _ESC_KEYS.get(seq, "")


@contextlib.contextmanager
def _raw_mode():
    if IS_WINDOWS:
        yield
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    mode = termios.tcgetattr(fd)
    mode[0] &= ~(termios.ICRNL | termios.IXON)                    # Enter reads as \r, Ctrl+S/Q are keys
    mode[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)    # key by key, we echo, Ctrl+C is a key
    termios.tcsetattr(fd, termios.TCSADRAIN, mode)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


class _View:
    """Redraws the prompt, the text and the suggestion list below it, then puts the cursor back."""

    def __init__(self, prompt):
        self.prompt, self.plen, self.cursor_row = prompt, visible_len(prompt), 0

    def draw(self, buf, pos, items=(), sel=0):
        width = shutil.get_terminal_size((80, 20)).columns
        parts = [f"\033[{self.cursor_row}A" if self.cursor_row else "", "\r\033[J", self.prompt, buf]
        end = self.plen + len(buf)  # ponytail: one column per character; wide CJK/emoji misplace the cursor
        if end and end % width == 0:
            parts.append(" \r")  # step past the terminal's pending wrap so row math stays exact
        for i, (text, desc) in enumerate(items):
            room = max(0, width - 4 - 22)
            line = f"{text:<20}  {desc[:room]}"
            parts.append("\n" + (rgb("❯ " + line, ACCENT[0], "1;") if i == sel else dim("  " + line)))
        row, col = divmod(self.plen + pos, width)
        up = end // width + len(items) - row
        parts.append((f"\033[{up}A" if up else "") + "\r" + (f"\033[{col}C" if col else ""))
        self.cursor_row = row
        sys.stdout.write("".join(parts))
        sys.stdout.flush()


class Wake(Exception):
    """Raised by read_line when `wake()` says something is waiting (a due job, a finished task)."""


def _key_ready(timeout):
    if IS_WINDOWS:
        deadline = time.monotonic() + timeout
        while not msvcrt.kbhit():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.02)
        return True
    return bool(select.select([sys.stdin.fileno()], [], [], timeout)[0])


def read_line(prompt, suggest, wake=None):
    """input() with live suggestions. `suggest(buf, pos)` returns a Suggestions. While the line is
    empty, `wake()` is polled and raises Wake when it's true, so background work can take the turn
    without clobbering anything the user is typing."""
    if not AVAILABLE:
        return input(readline_safe(prompt))
    view, buf, pos, sel, hidden, back = _View(prompt), "", 0, 0, False, 0
    keys = _windows_keys() if IS_WINDOWS else _unix_keys()

    def finish(text):
        view.draw(text, len(text))
        sys.stdout.write("\n")
        sys.stdout.flush()

    with _raw_mode():
        while True:
            hint = NONE if hidden else suggest(buf, pos)
            sel = min(sel, max(len(hint.items) - 1, 0))
            view.draw(buf, pos, hint.items, sel)
            while wake and not _key_ready(0.25):
                if not buf and wake():
                    view.draw("", 0)
                    sys.stdout.write("\r\033[J")  # remove the empty prompt; it's redrawn afterwards
                    sys.stdout.flush()
                    raise Wake()
            key = next(keys)
            if hint.items and key in ("\t", "\r", "\n"):
                pick = hint.items[sel][0]
                buf, pos = buf[:hint.start] + pick + buf[pos:], hint.start + len(pick)
                if hint.submit and key != "\t":
                    key = "\r"
                else:
                    if not buf[pos:pos + 1].isspace():
                        buf, pos = buf[:pos] + " " + buf[pos:], pos + 1
                    continue
            if key in ("\r", "\n"):
                finish(buf)
                if buf.strip() and (not history or history[-1] != buf):
                    history.append(buf)
                return buf
            if key == "\x03":
                finish(buf)
                raise KeyboardInterrupt
            if key in ("\x04", "\x1a") and not buf:
                finish(buf)
                raise EOFError
            if key in ("up", "down") and hint.items:
                sel = (sel + (1 if key == "down" else -1)) % len(hint.items)
                continue
            if key in ("up", "down") and history:
                back = max(0, min(len(history), back + (1 if key == "up" else -1)))
                buf = history[-back] if back else ""
                pos = len(buf)
                continue
            hidden = key == "esc"
            if key in ("\x08", "\x7f") and pos:
                buf, pos = buf[:pos - 1] + buf[pos:], pos - 1
            elif key == "delete":
                buf = buf[:pos] + buf[pos + 1:]
            elif key == "left":
                pos = max(0, pos - 1)
            elif key == "right":
                pos = min(len(buf), pos + 1)
            elif key in ("home", "\x01"):
                pos = 0
            elif key in ("end", "\x05"):
                pos = len(buf)
            elif key == "\x15":  # Ctrl+U clears the line
                buf, pos = "", 0
            elif len(key) == 1 and key >= " ":
                buf, pos, sel = buf[:pos] + key + buf[pos:], pos + 1, 0
