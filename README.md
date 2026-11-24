# ollama-code

A small Claude-Code-style coding agent for your terminal, powered by a local model through
Ollama's `/api/chat` API. The model can read, search and edit files and run commands in the
folder you start it in, and it asks you before anything that changes your files or runs a command.

Standard library only, no dependencies. Works on Windows (cmd, PowerShell, Windows Terminal,
Git Bash) and Linux (macOS too).

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running, with a model that supports tool calling:
  `ollama pull glm-4.7-flash`

## Install and run

Install it as an `ollama-code` command, then start it from the project folder you want it to work in:

```bash
uv tool install --editable /path/to/ollama-code   # or: pip install -e /path/to/ollama-code
cd your-project
ollama-code
```

`--editable` means edits to this folder take effect without reinstalling.

Without installing:

```bash
python3 /path/to/ollama-code/run.py        # Linux / macOS
py path\to\ollama-code\run.py              # Windows
```

One-shot, non-interactive (like `claude -p`):

```bash
ollama-code -p "explain what main.py does"
ollama-code --yolo -p "fix the failing test in tests/test_api.py"
```

## Options

| Flag | Env var | Default | What it does |
|---|---|---|---|
| `-m, --model` | `OLLAMA_CODE_MODEL` | `glm-4.7-flash` | Ollama model |
| `--host` | `OLLAMA_CODE_HOST` | `http://localhost:11434` | Ollama server |
| `--ctx` | `OLLAMA_CODE_CTX` | `0` | Context window to request. `0` uses whatever Ollama has loaded; any other size makes Ollama reload the model |
| `--no-think` | | thinking on | Turn off the model's reasoning step (faster) |
| `--hide-thinking` | | shown | Don't print the reasoning as it streams |
| `--yolo` | | off | Run edits and commands without asking |
| `--auto` | | off | Auto mode: no approval prompts at all, and the model decides instead of asking you |
| `-p PROMPT` | | | Run one prompt and exit |
| `-c, --continue` | | | Continue the most recent session from this folder |
| `-r, --resume [ID]` | | | Resume a session by id (or its first characters), or pick from a list |
| | `OLLAMA_CODE_HOME` | `~/.ollama-code` | Where sessions are saved |
| | `OLLAMA_CODE_TTY` | auto | `1` forces the animated terminal display, `0` turns it off |

## In the app

| Input | Action |
|---|---|
| `/help` | List commands |
| `/clear` | Start a fresh conversation (the old one stays saved) |
| `/compact` | Summarize the conversation to free context |
| `/usage` | Tokens in and out, speed and time for this session |
| `/resume [n\|id]` | List saved sessions, or resume one |
| `/model [name]` | Show or switch the model |
| `/think` | Toggle showing the model's reasoning (shown by default) |
| `/yolo` | Toggle asking before edits and commands |
| `/mcp` | List MCP servers, their status and tools |
| `/exit` | Quit (also Ctrl+D; Ctrl+Z then Enter on Windows) |
| `!command` | Run a shell command yourself; output goes into the conversation |
| `@path` | Attach a file. Type `@` for suggestions, or drag a file into the terminal (absolute paths attach too). Images go to vision models |
| line ending in `\` | Continue typing on the next line |
| Esc or Ctrl+C | Stop the current response (and the command it's running). Keys typed while the model works wait at the next prompt |
| Shift+Tab, `/auto` | Switch auto mode on/off. The prompt shows `⏵⏵ auto` while it's on |

Typing `/` shows matching commands as you type (arrows to pick, Tab to complete, Enter to run, Esc to hide).

## Claude Code skills and agents

Skills and agents you set up for Claude Code work here too. They're read from `./.claude`, `~/.claude`
(or `$CLAUDE_CONFIG_DIR`) and the plugins enabled in `~/.claude/settings.json`; `--no-claude` turns this off.

- **Skills** (`skills/<name>/SKILL.md`): the model loads one with its `use_skill` tool when a task matches,
  or you run one yourself with `/<skill> [request]` (they show up in the `/` suggestions). `/skills` lists them.
- **Agents** (`agents/<name>.md`): the model hands a task to one with `run_agent`. The agent runs in its own
  fresh conversation with the file's body as its system prompt, and its `tools:` list limits what it can use
  (`Read`, `Edit`, `Bash`, `Grep`, `Glob`, `mcp__server__*`…; tools ollama-code lacks, like WebFetch, are dropped).
  Only its final report comes back. `/agents` lists them.

Every agent runs on the current Ollama model; a `model:` line (sonnet, opus…) is ignored.

Agents can also run **in the background**: the model passes `background: true`, keeps working with you, and
the agent's report arrives as a message when it's done. Background agents can't ask for approval, so tools that
need it are refused unless you use `--yolo` or answered "always" for that tool earlier. `/tasks` lists them,
`/tasks stop <n|all>` stops them.

## Web search and fetch

Built in, no extra installs. The model searches with `web_search` and reads pages with `web_fetch`, which
returns the page's readable text (the `<main>`/article part when there is one) plus its links. JavaScript
doesn't run, so pages built entirely in the browser come back thin.

- Search runs without asking. Fetch asks once per website; "always" approves that site for the session.
- Search backend, first one configured wins:

| Env var | Backend |
|---|---|
| `OLLAMA_API_KEY` | Ollama's web search API (free key at ollama.com/settings/keys). Most reliable |
| `OLLAMA_CODE_SEARXNG` | Your SearXNG instance, e.g. `http://localhost:8888` (JSON format enabled) |
| (none) | DuckDuckGo, no key. Fine for occasional searches; bursts can trigger its bot check for a while |

Claude Code agents that list `WebSearch` / `WebFetch` in their `tools:` get these tools.

## Long-running commands

Commands never lock up the app. When the model's `run_command` is still going after 30 seconds (or it asks for
`background` straight away, e.g. for a dev server), it keeps running in the background, the model carries on, and
the result arrives as a message when the command ends. The model can check one with `shell_output` or end it with
`stop_shell`. Your own `!commands` move to the background after 5 seconds, so you get the prompt back.
`/shells` lists them, `/shells <n>` shows one's output, `/shells stop <n|all>` ends them. Background commands are
stopped when ollama-code exits. In `-p` mode commands simply run to completion (up to 10 minutes).

## Loops, schedules and goals

| Command | What it does |
|---|---|
| `/loop [interval] <prompt>` | Run a prompt now and then every interval (`30s`, `5m`, `1h30m`; default `10m`). `/loop` lists, `/loop stop [n]` ends |
| `/schedule in 10m <prompt>` | Run once after a delay. Also `at 14:30` / `at 9am`, and `every 2h` |
| `/schedule` | List jobs; `/schedule cancel <n\|all>` removes them |
| `/goal <objective>` | Keep working turn after turn until the model calls `goal_complete` (up to 20 rounds; Ctrl+C pauses). `/goal continue`, `/goal clear` |

Prompts can be anything you'd type, including skills: `/loop 30m /impeccable audit the dashboard`.
Jobs fire when the prompt is empty (never over something you're typing) and only while ollama-code is
open; for jobs that must survive a restart, run `ollama-code -p "..."` from cron or Task Scheduler.

## MCP servers

Stdio MCP servers in `.mcp.json` (project) or `~/.ollama-code/mcp.json` (user) start with the app, and their
tools are offered to the model. Same format as Claude Code:

```json
{"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}}}
```

MCP tools ask before running unless the server marks them read-only. HTTP/SSE servers aren't supported yet.

If the folder has an `AGENTS.md`, `CLAUDE.md` or `OLLAMA.md`, its contents are added to the
system prompt as project instructions.

## What you see

```
ollama-code v0.3.0 · http://localhost:11434
model glm-4.7-flash · loaded · context 202,752
> fix the average in calc.py
✻ Thinking
  The user wants me to fix calc.py. Let me read it first.
● Read(calc.py)
  ⎿  Read 6 lines
● Update(calc.py)
    - return sum(nums) / len(nums) + 1
    + return sum(nums) / len(nums)
  ⎿  Edited calc.py (1 replacement(s))
● Run(py calc.py)
  ⎿  4.0
● Fixed: the extra `+ 1` was inflating the result.
✻ 1m 14s · ↑ 4,210 in · ↓ 402 out · 9.1 tok/s · context 1,602 / 202,752
```

- At startup it says whether the model is already in memory. If not, it starts loading it in
  the background while you type.
- The model's reasoning streams in dim italics as it is generated.
- Whenever output pauses, an animated status line shows what is happening, for how long, and
  how many tokens have streamed at what speed, for example
  `✻ Loading glm-4.7-flash into memory… (14s · Ctrl+C to interrupt)` or
  `✻ Thinking… (41s · ↓ 230 tokens · 9.8 tok/s · Ctrl+C to interrupt)`.
  Ollama sends nothing while it loads the model, reads the prompt or writes a tool call, so
  these pauses can be long on a slow machine.
- Each tool call shows its result: command output, files or matches found, and a diff for edits.
- After each reply: time taken, tokens in (prompt) and out (generated), speed, and how full the
  context is. `/usage` shows the same totals for the whole session.
- Ctrl+C stops the response right away, even while Ollama is still silent, and disconnects so
  Ollama stops generating.

## Sessions

Every conversation is saved after each turn to `~/.ollama-code/sessions/<id>.json`, including
turns that were interrupted. Resume one with:

- `ollama-code -c`: the most recent session from the current folder
- `ollama-code -r`: pick from a list (`-r 20261121-0112` resumes by id)
- `/resume` inside the app, then type the number

Sessions contain the files and command output the model saw, so treat the folder as private.

## Tools the model can use

`read_file`, `list_files` and `grep` run freely. `write_file`, `edit_file` and `run_command`
show you what they will do and ask first. Answer `a` to allow a tool for the rest of the session.
Commands run in `cmd.exe` on Windows and `/bin/sh` on Linux, and the model is told which.

## Notes

- **Speed:** a 30B model on a small GPU runs mostly on CPU. Expect slow first responses
  while the model loads. `--no-think` and a smaller `--ctx` make it faster. If another client
  (such as Claude Code through `ollama launch`) is using the same model, requests wait for it.
- **Context:** the footer warns when the context is 80% full. Use `/compact` or `/clear`
  before it fills; Ollama silently drops the oldest text when it overflows.
- **Git Bash:** detected automatically, so animation and permission prompts work there too.
  If your terminal still shows no animation, set `OLLAMA_CODE_TTY=1`.
- **Ollama on another machine or WSL:** Ollama only listens on localhost by default. Set
  `OLLAMA_HOST=0.0.0.0` on the server and pass `--host http://<server-ip>:11434`.

## Layout

| File | What's in it |
|---|---|
| `ollama_code/cli.py` | Prompt loop, slash commands, command-line options |
| `ollama_code/agent.py` | The agent loop, tool-call display, usage stats |
| `ollama_code/tools.py` | The tools the model can call |
| `ollama_code/client.py` | Streaming from Ollama, model loading checks |
| `ollama_code/terminal.py` | Colors, terminal detection, the live status line |
| `ollama_code/sessions.py` | Saving and resuming conversations |

## Test

```bash
python test_ollama_code.py    # or: pytest
```

Runs the tools, the full agent loop, sessions and Ctrl+C handling against fake Ollama servers,
so no model is needed.
