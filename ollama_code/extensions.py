"""Claude Code skills and agents, reused as-is.

Found in ./.claude (project), ~/.claude (user, or $CLAUDE_CONFIG_DIR) and the plugins enabled in
~/.claude/settings.json; earlier places win on name clashes. Plugin items are named "plugin:name".

- Skills (skills/<name>/SKILL.md): the model loads one with the use_skill tool, or you run `/name args`.
- Agents (agents/<name>.md): the model delegates with run_agent; the file body becomes the sub-agent's
  system prompt and its `tools:` list (Claude names like Read, Bash, mcp__x__*) limits what it may call.
"""
import fnmatch
import json
import os
import re
from pathlib import Path

CLAUDE_HOME = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
CATALOG_CHARS = 12_000  # past this, the system prompt lists names only, to spare small context windows

# Claude Code tool names -> ours. Tools we don't have (WebFetch, Task, ...) are dropped.
CLAUDE_TOOLS = {"Read": "read_file", "Write": "write_file", "Edit": "edit_file", "MultiEdit": "edit_file",
                "NotebookEdit": "edit_file", "Glob": "list_files", "LS": "list_files", "Grep": "grep",
                "Bash": "run_command", "PowerShell": "run_command"}

skills, agents = {}, {}  # name -> {"name", "description", "path", "tools"}


def frontmatter(text):
    """(fields, body) from a markdown file's --- frontmatter. Handles the YAML these files use:
    `key: value`, quoted values, `|`/`>` blocks and `- item` lists. Nested maps are skipped."""
    m = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n?", text, re.S)
    if not m:
        return {}, text
    fields, key = {}, None
    for line in m.group(1).splitlines():
        top = re.match(r"([A-Za-z_][\w-]*):\s*(.*)$", line)
        if top:
            key, value = top.group(1), top.group(2).strip()
            fields[key] = "" if value in ("|", ">", "|-", ">-") else value.strip("'\"")
        elif key and line.strip().startswith("- "):
            fields[key] = (fields[key] + ", " if fields[key] else "") + line.strip()[2:].strip("'\"")
        elif key and line.startswith((" ", "\t")) and not re.match(r"\s+[\w-]+:\s", line):
            fields[key] = (fields[key] + " " + line.strip()).strip()
    return fields, text[m.end():]


def _plugin_roots():
    try:
        enabled = json.loads((CLAUDE_HOME / "settings.json").read_text(encoding="utf-8")).get("enabledPlugins", {})
        installed = json.loads((CLAUDE_HOME / "plugins" / "installed_plugins.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, AttributeError):
        return []
    roots = []
    for key, on in enabled.items():
        for install in installed.get("plugins", {}).get(key, [])[:1] if on else []:
            roots.append((Path(install["installPath"]), key.split("@")[0] + ":"))
    return roots


def _add(table, name, path, fields):
    if name and name not in table:
        table[name] = {"name": name, "description": " ".join(fields.get("description", "").split()),
                       "path": path, "tools": fields.get("tools", "")}


def load():
    skills.clear()
    agents.clear()
    for root, prefix in [(Path.cwd() / ".claude", ""), (CLAUDE_HOME, "")] + _plugin_roots():
        for path in sorted(root.glob("skills/*/SKILL.md")):
            fields, _ = _read(path)
            _add(skills, prefix + (fields.get("name") or path.parent.name), path, fields)
        for path in sorted(root.glob("agents/*.md")):
            fields, _ = _read(path)
            _add(agents, prefix + (fields.get("name") or path.stem), path, fields)
    return len(skills), len(agents)


def _read(path):
    try:
        return frontmatter(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return {}, ""


def catalog():
    """The part of the system prompt that tells the model what it can use."""
    if not (skills or agents):
        return ""

    def listing(table):
        full = "\n".join(f"- {n}: {i['description'][:90]}" for n, i in table.items())
        return full if len(full) <= CATALOG_CHARS // 2 else ", ".join(table)

    text = ""
    if skills:
        text += ("\n\nSkills: specialised instructions. When a task matches one, call use_skill with its name "
                 "first and follow what it says.\n" + listing(skills))
    if agents:
        text += ("\n\nAgents: specialists you can hand a self-contained task to with run_agent. They start "
                 "with no memory of this conversation, so put everything they need in the task.\n" + listing(agents))
    return text


def schemas():
    out = []
    if skills:
        out.append(_schema("use_skill", "Load a skill's full instructions by name (see the Skills list).",
                           {"name": {"type": "string", "description": "Skill name"}}, ["name"]))
    if agents:
        out.append(_schema("run_agent", "Run a specialist agent on a task and get its final report back.",
                           {"agent": {"type": "string", "description": "Agent name (see the Agents list)"},
                            "task": {"type": "string", "description": "Complete, self-contained instructions"},
                            "background": {"type": "boolean", "description":
                                           "Run it in the background and carry on; its report arrives later "
                                           "as a message. Use for long, independent work."}},
                           ["agent", "task"]))
    return out


def _schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


def _unknown(kind, name, table):
    close = [n for n in table if name.lower() in n.lower() or n.lower() in name.lower()][:8]
    return f"Error: no {kind} named '{name}'." + (f" Did you mean: {', '.join(close)}?" if close else "")


def skill_text(name, arguments=""):
    item = skills.get(name)
    if not item:
        return _unknown("skill", name, skills)
    _, body = _read(item["path"])
    body = body.replace("$ARGUMENTS", arguments)
    return (f"Skill '{name}'. Its files live in {item['path'].parent}; read any it refers to with read_file.\n\n"
            + body.strip())


def agent_prompt(name):
    """(system prompt, its `tools:` spec) for an agent. LookupError if there's no such agent."""
    item = agents.get(name)
    if not item:
        raise LookupError(_unknown("agent", name, agents))
    _, body = _read(item["path"])
    return body.strip(), item["tools"]


def allowed_tools(spec, available):
    """Which of `available` tool names an agent's `tools:` field permits. Empty or * means all."""
    wanted = [t.strip().strip("'\"") for t in spec.strip("[]").split(",") if t.strip()]
    if not wanted or "*" in wanted:
        return list(available)
    ours = {CLAUDE_TOOLS[t] for t in wanted if t in CLAUDE_TOOLS}
    return [n for n in available if n in ours or any(fnmatch.fnmatchcase(n, t) for t in wanted)]
