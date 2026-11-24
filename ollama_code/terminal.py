"""Terminal output: colors, terminal detection and the live status line."""
import contextlib
import itertools
import os
import re
import shutil
import sys
import threading
import time

try:
    import readline  # noqa: F401  (line editing + history on Linux/macOS)
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

IS_WINDOWS = os.name == "nt"


def _is_msys_pty(stream):
    """True in Git Bash / MSYS2 / Cygwin terminals (mintty). They give Windows programs a named pipe
    instead of a console, so isatty() says False even though a person is watching."""
    if not IS_WINDOWS:
        return False
    try:
        import ctypes
        import msvcrt

        class FileNameInfo(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("name", ctypes.c_wchar * 260)]

        info = FileNameInfo()
        handle = ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno()))
        if not ctypes.windll.kernel32.GetFileInformationByHandleEx(handle, 2, ctypes.byref(info), ctypes.sizeof(info)):
            return False  # 2 = FileNameInfo; fails for consoles and files, which is fine
        name = info.name[: info.length // 2]
    except (AttributeError, OSError, ValueError):
        return False
    return ("msys-" in name or "cygwin-" in name) and "-pty" in name


def _is_terminal(stream):
    forced = os.environ.get("OLLAMA_CODE_TTY")  # escape hatch for terminals we can't detect
    if forced in ("0", "1"):
        return forced == "1"
    try:
        return stream.isatty() or _is_msys_pty(stream)
    except (AttributeError, ValueError):
        return False


if IS_WINDOWS:
    os.system("")  # turns on ANSI escape-code handling in the Windows console
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    try:
        if IS_WINDOWS and not _stream.isatty():
            # Pipes, files and mintty (Git Bash) get UTF-8. Python's default there is cp1252, which turns
            # ●, ↑ and ✻ into "?". Real consoles are unaffected; Python writes Unicode to them directly.
            _stream.reconfigure(encoding="utf-8")
        _stream.reconfigure(errors="replace")  # never crash on a character the console can't show
    except (AttributeError, ValueError):
        pass
TTY = _is_terminal(sys.stdout)
INTERACTIVE = _is_terminal(sys.stdin)
COLOR = TTY and "NO_COLOR" not in os.environ


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def dim(t): return paint(t, "2")
def faint(t): return paint(t, "2;3")  # dim italic, for the model's reasoning
def bold(t): return paint(t, "1")
def red(t): return paint(t, "31")
def green(t): return paint(t, "32")
def yellow(t): return paint(t, "33")
def cyan(t): return paint(t, "36")
def magenta(t): return paint(t, "35")


# ponytail: assumes truecolor; Windows Terminal, iTerm, modern VTE all have it. Old conhost shows nearest match.
ACCENT = (0, 200, 255), (190, 90, 255)  # cyan -> violet, the brand gradient
ANSI_RE = re.compile("\033\\[[0-9;]*m")


def visible_len(text):
    return len(ANSI_RE.sub("", text))


def rgb(text, color, extra=""):
    return f"\033[{extra}38;2;{color[0]};{color[1]};{color[2]}m{text}\033[0m" if COLOR else text


def _mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def gradient(text, start=ACCENT[0], end=ACCENT[1], extra=""):
    """Color each character along a left-to-right gradient."""
    if not COLOR or len(text) < 2:
        return rgb(text, start, extra)
    return "".join(rgb(ch, _mix(start, end, i / (len(text) - 1)), extra) for i, ch in enumerate(text))


def shimmer(text, frame, base=(150, 150, 160), glow=(255, 255, 255), width=4):
    """A bright band sweeping across the text, one character per frame."""
    if not COLOR:
        return text
    pos = frame % (len(text) + width * 3) - width
    return "".join(rgb(ch, _mix(base, glow, max(0.0, 1 - abs(i - pos) / width))) for i, ch in enumerate(text))


def readline_safe(prompt):
    """Mark color codes as zero-width so readline keeps the cursor in the right place."""
    return re.sub("(\033\\[[0-9;]*m)", "\001\\1\002", prompt) if HAS_READLINE else prompt


class Screen:
    """Owns the terminal, so the live status line never garbles streamed text.

    The status line ("✻ Thinking… (12s · Ctrl+C to interrupt)") is drawn by a background thread
    whenever output has paused, and erased before anything else is written. It only runs on a
    real terminal, never when output is piped or logged.
    """
    FRAMES = "·✢✳✶✻✽✻✶✳✢"

    def __init__(self, live):
        self.lock = threading.Lock()
        self.status = None              # activity shown on the status line; None hides it
        self.clock = time.monotonic()   # when the current turn started, for the elapsed counter
        self.drawn = False              # is the status line on screen right now?
        self.line_start = True          # is the cursor at the start of a line?
        self.last_write = 0.0
        self.turn_tokens = 0            # tokens streamed so far this turn
        self.step_tokens, self.step_first, self.step_last = 0, 0.0, 0.0  # for the live tok/s
        self.live = live
        self.pause_keys = None          # set by the Esc watcher: a context that hands the keyboard back
        self.box, self.box_rows, self.box_screen_rows = None, 0, 0  # docked input box, see dock()
        self.pinned = None              # label of an animated header line above the cursor, e.g. "Thinking"
        self.pin_start = 0.0
        self.rows_below = self.col = 0  # rows printed since the pinned header, and the cursor column
        if live:
            threading.Thread(target=self._animate, daemon=True).start()

    def write(self, text):
        with self.lock:
            self._erase()
            sys.stdout.write(text)
            sys.stdout.flush()
            if self.pinned:
                self._count_rows(text)
            if text:
                self.line_start = text.endswith("\n")
            self.last_write = time.monotonic()

    def _count_rows(self, text):
        """Track how far below the pinned header the cursor is, including soft-wrapped lines."""
        width = shutil.get_terminal_size((80, 20)).columns
        for ch in ANSI_RE.sub("", text):
            if ch == "\n":
                self.rows_below, self.col = self.rows_below + 1, 0
            elif ch == "\r":
                self.col = 0
            else:
                self.col += 1
                if self.col > width:  # ponytail: counts wide chars (CJK, emoji) as one column
                    self.rows_below, self.col = self.rows_below + 1, 1

    def pin(self, label):
        """Animate the line just printed (cursor must be at the start of the next line)."""
        with self.lock:
            if self.live:
                self.pinned, self.pin_start, self.rows_below, self.col = label, time.monotonic(), 0, 0

    def unpin(self, final):
        """Stop animating the pinned line and leave `final` in its place."""
        with self.lock:
            if self.pinned:
                self._draw_pinned(final)
                self.pinned = None
                sys.stdout.flush()

    def _draw_pinned(self, text):
        up = self.rows_below + 1
        if up < shutil.get_terminal_size((80, 20)).lines - self.box_rows:  # still on screen, not scrolled away
            sys.stdout.write(f"\0337\033[{up}A\r\033[K{text}\0338")  # save cursor, jump up, redraw, restore

    # --- the docked input box: shown at the bottom while the model works ---------------------------
    BOX_ROWS = 2  # a status rule and the input line

    def dock(self, render):
        """Keep the bottom rows for an input box: output scrolls above it (a terminal scroll region),
        the box stays put. `render(width)` returns the box's lines. False if the terminal is too small."""
        rows = shutil.get_terminal_size((80, 24)).lines
        if not self.live or rows < 10:
            return False
        with self.lock:
            self._erase()
            if not self.line_start:
                sys.stdout.write("\n")
                self.line_start = True
            n = self.BOX_ROWS
            # make room so the cursor isn't inside the box, then fence the box off; DECSTBM homes the
            # cursor, hence the save/restore around it
            sys.stdout.write("\n" * n + f"\033[{n}A\0337\033[1;{rows - n}r\0338")
            self.box, self.box_rows, self.box_screen_rows = render, n, rows
            self._draw_box()
            sys.stdout.flush()
        return True

    def undock(self):
        with self.lock:
            if not self.box:
                return
            self._erase()
            rows = self.box_screen_rows
            sys.stdout.write(f"\0337\033[r\0338\0337\033[{rows - self.box_rows + 1};1H\033[J\0338")
            self.box, self.box_rows = None, 0
            sys.stdout.flush()

    def refresh_box(self):
        with self.lock:
            self._draw_box()
            sys.stdout.flush()

    def _draw_box(self):
        if not self.box:
            return
        size = shutil.get_terminal_size((80, 24))
        if size.lines != self.box_screen_rows:  # window resized: move the fence
            self.box_screen_rows = size.lines
            sys.stdout.write(f"\0337\033[1;{size.lines - self.box_rows}r\0338")
        lines = self.box(size.columns)
        sys.stdout.write("\0337" + "".join(f"\033[{size.lines - self.box_rows + 1 + i};1H\033[2K{line}"
                                           for i, line in enumerate(lines[:self.box_rows])) + "\0338")

    def pinned_frame(self, frame):
        spin = self.FRAMES[(frame // 2) % len(self.FRAMES)]
        hue = _mix(*ACCENT, abs((frame % 40) - 20) / 20)
        return (f"{rgb(spin, hue, '1;')} {shimmer(self.pinned + '…', frame, base=(120, 140, 200))} "
                + dim(f"{int(time.monotonic() - self.pin_start)}s"))

    def new_turn(self):
        with self.lock:
            self.clock, self.turn_tokens, self.step_tokens = time.monotonic(), 0, 0

    def new_step(self):
        with self.lock:
            self.step_tokens = 0

    def note_token(self):
        """Ollama streams one token per chunk; count them for the live status line."""
        now = time.monotonic()
        with self.lock:
            if not self.step_tokens:
                self.step_first = now
            self.step_tokens += 1
            self.turn_tokens += 1
            self.step_last = now

    def progress(self):
        parts = [f"{int(time.monotonic() - self.clock)}s"]
        if self.turn_tokens:
            parts.append(f"↓ {self.turn_tokens:,} tokens")
        if self.step_tokens > 1 and self.step_last > self.step_first:
            parts.append(f"{(self.step_tokens - 1) / (self.step_last - self.step_first):.1f} tok/s")
        return " · ".join(parts + ["Esc to interrupt" if self.pause_keys else "Ctrl+C to interrupt"])

    def set_status(self, text):
        with self.lock:
            self.status = text
            if text is None:
                self._erase()
                sys.stdout.flush()

    def ask(self, prompt, reader=None):
        """input() (or `reader`) that hides the status line and starts on a fresh line."""
        with self.lock:
            self.status = None
            self._erase()
            if not self.line_start:
                sys.stdout.write("\n")
            self.pinned = None  # the prompt would scroll it anyway
            sys.stdout.flush()
        try:
            with self.pause_keys() if self.pause_keys else contextlib.nullcontext():
                return reader(prompt) if reader else input(readline_safe(prompt))
        finally:
            with self.lock:
                self.line_start, self.last_write = True, time.monotonic()

    def _erase(self):
        if self.drawn:
            sys.stdout.write("\r\033[K")
            self.drawn = False

    def tick(self, frame):
        """Draw one animation frame if output has paused. Called ~16 times a second."""
        with self.lock:
            if self.pinned:
                self._draw_pinned(self.pinned_frame(frame))
                sys.stdout.flush()
            if self.box and frame % 8 == 0:  # twice a second: follow window resizes, refresh the box
                self._draw_box()
                sys.stdout.flush()
            idle = time.monotonic() - self.last_write
            if self.status is None or idle < 0.5 or (not self.line_start and idle < 2):
                return
            if not self.line_start:  # output stopped mid-line, e.g. while the model writes a tool call
                sys.stdout.write("\n")
                self.line_start = True
                if self.pinned:
                    self._count_rows("\n")
            spin = self.FRAMES[(frame // 2) % len(self.FRAMES)]
            hue = _mix(*ACCENT, abs((frame % 40) - 20) / 20)  # spinner breathes cyan <-> violet
            sys.stdout.write(f"\r\033[K{rgb(spin, hue, '1;')} {shimmer(self.status[:40] + '…', frame)} "
                             + dim(f"({self.progress()})"))
            sys.stdout.flush()
            self.drawn = True

    def _animate(self):
        for frame in itertools.count():
            time.sleep(0.06)
            self.tick(frame)


screen = Screen(live=TTY)


def out(text="", end="\n"):
    screen.write(f"{text}{end}")


def new_line():
    if not screen.line_start:
        out()
