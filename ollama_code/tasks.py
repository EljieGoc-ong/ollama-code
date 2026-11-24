"""Background sub-agents: run in a thread, report back as a message when done. /tasks lists them."""
import itertools
import threading
import time

from .agent import Stopped, agent_loop, make_subagent

tasks = {}  # id -> Task
_ids = itertools.count(1)
_lock = threading.Lock()


class Task:
    def __init__(self, agent, task):
        self.id, self.agent, self.task = next(_ids), agent, task
        self.status, self.report, self.delivered = "running", "", False
        self.started, self.ended = time.monotonic(), None
        self.stop = threading.Event()


def start(state, agent, task):
    """Launch `agent` on `task` in the background. Returns the tool result for the model."""
    try:
        child = make_subagent(state, agent, task)
    except LookupError as e:
        return str(e)
    job = Task(agent, task)
    child.background = job.stop
    child.messages = list(child.messages)
    tasks[job.id] = job

    def work():
        try:
            report, status = agent_loop(child) or "(the agent finished without a report)", "done"
        except Stopped:
            report, status = "(stopped by the user)", "stopped"
        except Exception as e:  # report every failure; a dead thread must not lose the task silently
            report, status = f"Error: {type(e).__name__}: {e}", "failed"
        with _lock:
            job.report, job.status, job.ended = report, status, time.monotonic()

    threading.Thread(target=work, daemon=True).start()
    return (f"Started background task #{job.id} ({agent}). Carry on with other work; its report will arrive "
            "as a message when it finishes. Don't wait or poll for it.")


def take_finished():
    """Finished tasks not yet reported to the model, oldest first."""
    with _lock:
        ready = [t for t in tasks.values() if t.status != "running" and not t.delivered]
        for t in ready:
            t.delivered = True
    return ready


def has_finished():
    return any(t.status != "running" and not t.delivered for t in tasks.values())


def stop(which):
    """Stop task number `which`, or 'all'. It ends at its next streamed chunk. Returns how many."""
    hit = [t for t in tasks.values() if t.status == "running" and (which == "all" or str(t.id) == which.lstrip("#"))]
    for t in hit:
        t.stop.set()
    return len(hit)


def describe(t):
    took = int((t.ended or time.monotonic()) - t.started)
    return f"#{t.id} {t.agent} · {t.status} · {took}s · {t.task[:70]}"


def notice(t):
    """The message that hands a finished task's report back to the model."""
    return (f"[Background task #{t.id} ({t.agent}) {t.status}. This is its report, not a message from the user.]\n"
            f"Task was: {t.task}\n\nReport:\n{t.report}\n\n"
            "Use this if it matters for what we're doing; otherwise just tell the user briefly what it found.")
