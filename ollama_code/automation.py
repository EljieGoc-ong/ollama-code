"""/loop, /schedule, /tasks and /goal: work that runs without a fresh prompt from the user."""
from . import schedule, tasks
from .terminal import ACCENT, bold, dim, green, out, red, rgb, yellow

MAX_GOAL_ROUNDS = 20  # model turns /goal spends before checking back with the user


def loop_command(arg):
    if not arg or arg == "list":
        return _list([j for j in schedule.jobs if j.kind == "loop"], "No loops running. Start one: /loop 5m <prompt>")
    if arg.split()[0] == "stop":
        which = (arg.split() + ["all"])[1]
        n = schedule.cancel(which, kind="loop") if which == "all" else schedule.cancel(which)
        return out(dim(f"Stopped {n} loop{'s' * (n != 1)}."))
    try:
        due, every, prompt = schedule.parse_loop(arg)
    except ValueError as e:
        return out(yellow(str(e)))
    job = schedule.add(due, every, prompt, kind="loop")
    out(dim(f"  ⎿  Loop #{job.id} runs now, then every {schedule._span(every)}. /loop stop ends it."))


def schedule_command(arg):
    if not arg or arg == "list":
        return _list(schedule.jobs, "Nothing scheduled. Try: /schedule at 9am summarize yesterday's commits")
    if arg.split()[0] == "cancel":
        n = schedule.cancel((arg.split() + ["all"])[1])
        return out(dim(f"Cancelled {n} job{'s' * (n != 1)}."))
    try:
        due, every, prompt = schedule.parse_schedule(arg)
    except ValueError as e:
        return out(yellow(str(e)))
    job = schedule.add(due, every, prompt)
    out(dim(f"  ⎿  Scheduled {schedule.describe(job)}"))
    out(dim("     Runs while ollama-code stays open."))


def _list(jobs, empty):
    if not jobs:
        return out(dim(empty))
    for job in jobs:
        out("  " + schedule.describe(job))


def tasks_command(arg):
    if arg.startswith("stop"):
        n = tasks.stop((arg.split() + ["all"])[1])
        return out(dim(f"Stopping {n} task{'s' * (n != 1)}; each ends at its next step."))
    if not tasks.tasks:
        return out(dim("No background agents. Ask the model to run an agent in the background."))
    for t in tasks.tasks.values():
        mark = {"running": yellow("◌"), "done": green("●"), "stopped": dim("■")}.get(t.status, red("✗"))
        out(f"  {mark} {tasks.describe(t)}")


def due_work():
    """Prompts that are ready to run now: (header line, prompt, from_user). Finished background tasks
    first, so their reports land before a scheduled job might need them."""
    for t in tasks.take_finished():
        yield green("● ") + bold(f"Background task #{t.id} ({t.agent}) {t.status}"), tasks.notice(t), False
    for job in schedule.take_due():
        yield rgb("⟳ ", ACCENT[0], "1;") + dim(f"{job.kind} #{job.id} (run {job.runs}): ") + job.prompt, job.prompt, True


def goal_command(state, arg, run_prompt):
    """/goal <objective> sets a goal and works on it; /goal shows it; /goal clear drops it."""
    if not arg:
        return out(dim(f"Goal: {state.goal}" if state.goal else "No goal set. /goal <objective> sets one."))
    if arg == "clear":
        state.goal = None
        return out(dim("Goal cleared."))
    if arg == "continue":
        if not state.goal:
            return out(dim("No goal to continue."))
        return pursue_goal(state, run_prompt, fresh=False)
    state.goal, state.goal_result = arg, None
    out(dim(f"  ⎿  Working until it's done: up to {MAX_GOAL_ROUNDS} rounds. Ctrl+C pauses."))
    pursue_goal(state, run_prompt, fresh=True)


def goal_prompt(goal, round_no, fresh):
    if fresh and round_no == 1:
        return (f"New goal: {goal}\n\nWork on it until it is fully done and verified (run the tests or the "
                "code to check). Keep going without asking me unless you're truly blocked. When it's done, call "
                "goal_complete with a summary; if you need me, call goal_complete with blocked=true and say what.")
    return (f"Keep going on the goal: {goal}\n(Round {round_no} of {MAX_GOAL_ROUNDS}.) Check what's still "
            "missing and do it. When it's fully done and verified, call goal_complete; if you're blocked on me, "
            "call it with blocked=true.")


def pursue_goal(state, run_prompt, fresh):
    for round_no in range(1, MAX_GOAL_ROUNDS + 1):
        goal = state.goal
        if round_no > 1:
            out(rgb("◎ ", ACCENT[1], "1;") + dim(f"goal · round {round_no}/{MAX_GOAL_ROUNDS}"))
        if not run_prompt(state, goal_prompt(goal, round_no, fresh)):
            return out(yellow("Goal paused. /goal continue picks it up again, /goal clear drops it."))
        if state.goal is None:
            summary, blocked = state.goal_result
            label = yellow("◎ Goal blocked: ") if blocked else green("◎ Goal complete: ")
            return out(label + summary)
    out(yellow(f"Goal still open after {MAX_GOAL_ROUNDS} rounds. /goal continue keeps going, /goal clear drops it."))
