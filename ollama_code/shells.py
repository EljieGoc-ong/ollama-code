"""Shell commands that keep running in the background, so a slow build or a dev server never
blocks the conversation. Output is collected as it arrives; finished commands are reported back."""
import atexit
import itertools
import os
import signal
import subprocess
import threading
import time

FOREGROUND_WAIT = 30      # seconds run_command waits before moving a command to the background
KEEP_OUTPUT = 200_000     # characters of output kept per command (the most recent ones)
IS_WINDOWS = os.name == "nt"

shells = {}               # id -> Shell
_ids = itertools.count(1)
_lock = threading.Lock()


class Shell:
    def __init__(self, command, owner):
        self.id, self.command, self.owner = next(_ids), command, owner  # owner: "model" or "user"
        self.output, self.dropped = "", 0   # text so far, and how many older characters were cut
        self.code, self.started, self.ended = None, time.monotonic(), None
        self.backgrounded = False            # moved to the background, so its end must be reported
        self.delivered = False
        self.done = threading.Event()
        self.proc = subprocess.Popen(
            command, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=not IS_WINDOWS,  # its own process group, so stop() takes children down too
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0)
        threading.Thread(target=self._collect, daemon=True).start()

    def _collect(self):
        fd = self.proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            with _lock:
                self.output += chunk.decode("utf-8", errors="replace")
                if len(self.output) > KEEP_OUTPUT:
                    cut = len(self.output) - KEEP_OUTPUT
                    self.output, self.dropped = self.output[cut:], self.dropped + cut
        self.code, self.ended = self.proc.wait(), time.monotonic()
        self.done.set()

    @property
    def running(self):
        return not self.done.is_set()

    def text(self, tail=None):
        with _lock:
            body = self.output if tail is None else self.output[-tail:]
            skipped = self.dropped + len(self.output) - len(body)
        body = body.replace("\r\n", "\n").rstrip()
        return (f"... [{skipped:,} earlier characters not shown]\n" if skipped else "") + body

    def result(self):
        """The same shape run_command has always returned: 'exit code N' then the output."""
        return f"exit code {self.code}\n{self.text()}".rstrip()

    def stop(self):
        if not self.running:
            return
        try:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                               capture_output=True, timeout=10)
            else:
                os.killpg(self.proc.pid, signal.SIGTERM)
        except (OSError, subprocess.SubprocessError):
            self.proc.kill()
        self.done.wait(5)

    def describe(self):
        took = int((self.ended or time.monotonic()) - self.started)
        state = "running" if self.running else f"exit {self.code}"
        return f"#{self.id} {state} · {took}s · {self.command[:80]}"


def start(command, owner="model"):
    shell = Shell(command, owner)
    shells[shell.id] = shell
    return shell


def find(which):
    return shells.get(int(str(which).lstrip("#"))) if str(which).lstrip("#").isdigit() else None


def take_finished():
    """Background commands that ended and haven't been reported yet, oldest first."""
    ready = [s for s in shells.values() if s.backgrounded and not s.running and not s.delivered]
    for s in ready:
        s.delivered = True
    return ready


def has_finished():
    return any(s.backgrounded and not s.running and not s.delivered for s in shells.values())


def stop(which):
    hit = [s for s in shells.values() if s.running and (which == "all" or s is find(which))]
    for s in hit:
        s.stop()
    return len(hit)


# ponytail: background commands die with ollama-code; start long-lived servers outside it if they must outlive it
atexit.register(lambda: stop("all"))
