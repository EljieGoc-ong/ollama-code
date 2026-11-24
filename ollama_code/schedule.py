"""Prompts that run later or repeatedly while ollama-code is open: /loop and /schedule.

ponytail: in-memory and in-process. Jobs run only while the app is open and are gone when it exits;
use cron / Task Scheduler with `ollama-code -p` for anything that must survive a restart.
"""
import datetime
import itertools
import re
import time

UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
DEFAULT_LOOP = 600  # /loop without an interval: every 10 minutes
MIN_INTERVAL = 10   # seconds; anything faster would just stack model calls back to back

jobs = []           # Job objects, in creation order
_ids = itertools.count(1)


class Job:
    def __init__(self, prompt, due, every=None, kind="schedule"):
        self.id, self.prompt, self.due, self.every, self.kind = next(_ids), prompt, due, every, kind
        self.runs = 0


def parse_duration(text):
    """'90s', '5m', '1h30m', '2d' -> seconds, or None if it isn't a duration."""
    parts = re.fullmatch(r"(?:\d+[smhd])+", text.strip().lower() or "x")
    if not parts:
        return None
    return sum(int(n) * UNITS[u] for n, u in re.findall(r"(\d+)([smhd])", text.lower()))


def _next_clock(text, now):
    """'14:30' or '9am' / '9:15pm' -> the next time the clock shows it, as an epoch."""
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text.strip().lower())
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    if m.group(3):
        hour = hour % 12 + (12 if m.group(3) == "pm" else 0)
    if hour > 23 or minute > 59:
        return None
    current = datetime.datetime.fromtimestamp(now)
    at = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if at <= current:
        at += datetime.timedelta(days=1)
    return at.timestamp()


def parse_schedule(text, now=None):
    """'in 10m <prompt>', 'at 14:30 <prompt>', 'every 1h <prompt>' -> (due, every, prompt). ValueError if not."""
    now = time.time() if now is None else now
    word, _, rest = text.strip().partition(" ")
    when, _, prompt = rest.strip().partition(" ")
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("say what to run, e.g. /schedule in 10m run the tests")
    word = word.lower()
    if word in ("in", "every"):
        seconds = parse_duration(when)
        if not seconds:
            raise ValueError(f"'{when}' isn't a duration like 30s, 10m, 1h or 1h30m")
        if word == "every":
            return now + max(seconds, MIN_INTERVAL), max(seconds, MIN_INTERVAL), prompt
        return now + seconds, None, prompt
    if word == "at":
        due = _next_clock(when, now)
        if due is None:
            raise ValueError(f"'{when}' isn't a time like 14:30 or 9am")
        return due, None, prompt
    raise ValueError("start with in, at or every, e.g. /schedule at 9am summarize yesterday's commits")


def parse_loop(text, now=None):
    """'/loop [interval] <prompt>' arguments -> (due, every, prompt). The first run is right away."""
    now = time.time() if now is None else now
    first, _, rest = text.strip().partition(" ")
    seconds = parse_duration(first)
    prompt = rest.strip() if seconds else text.strip()
    if not prompt:
        raise ValueError("say what to repeat, e.g. /loop 5m check the build")
    return now, max(seconds or DEFAULT_LOOP, MIN_INTERVAL), prompt


def add(due, every, prompt, kind="schedule"):
    job = Job(prompt, due, every, kind)
    jobs.append(job)
    return job


def take_due(now=None):
    """Jobs whose time has come, oldest first. Repeating jobs move to their next slot (missed slots
    collapse into one run, so a long turn never causes a burst); one-shots are removed."""
    now = time.time() if now is None else now
    ready = sorted((j for j in jobs if j.due <= now), key=lambda j: j.due)
    for job in ready:
        job.runs += 1
        if job.every:
            while job.due <= now:
                job.due += job.every
        else:
            jobs.remove(job)
    return ready


def has_due(now=None):
    now = time.time() if now is None else now
    return any(j.due <= now for j in jobs)


def cancel(which, kind=None):
    """Cancel job number `which`, or 'all' (optionally only of one kind). Returns how many."""
    doomed = [j for j in jobs if (which == "all" and kind in (None, j.kind)) or str(j.id) == which.lstrip("#")]
    for job in doomed:
        jobs.remove(job)
    return len(doomed)


def _span(seconds):
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size and seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def describe(job, now=None):
    now = time.time() if now is None else now
    wait = max(0, int(job.due - now))
    when = f"every {_span(job.every)}" if job.every else \
        datetime.datetime.fromtimestamp(job.due).strftime("at %H:%M")
    return f"#{job.id} {job.kind} · {when} · next in {_span(wait) if wait >= 60 else f'{wait}s'} · {job.prompt}"
