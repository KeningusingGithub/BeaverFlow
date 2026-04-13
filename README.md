# BeaverFlow

BeaverFlow is a minimal agent workflow framework built around two ideas:

1. Each agent is defined as a small Python `Agent(...)` object.
2. Each agent runs inside its own namespace under one shared workspace.

The core split is intentional:

- `agent.py` runs one agent and enforces `task`, `acceptance`, and optional `mindset`.
- `agent_workflow.py` composes multiple agents with a small workflow DSL such as `A->B`, `A and B`, and `loop(A->B, 3)`.

## Repository layout

```text
BeaverFlow/
  agent.py
  agent_workflow.py
  agent_workflow_sample.py
  agent_workflow_sample2.py
  README.md
  LICENSE
  .gitignore
  requirements.txt
  pyproject.toml
```

## What is stable in this repo

- `task` and `acceptance` stay separate.
- `mindset` is optional. If omitted, `agent.py` uses its built-in default engineering mindset.
- `loop(...)` is mechanical repetition. It does not stop early.
- Workflow-side agent definitions use class style only:

```python
A = Agent(
    name="A",
    script="./agent.py",
    workspace=WORKSPACE,
    task=A_TASK,
    acceptance=A_ACCEPTANCE,
)
```

Required fields:

- `name`
- `script`
- `workspace`
- `task`
- `acceptance`

Optional fields include `mindset`, `max_rounds`, `reset`, `dry_run`, `ai_model`, `codex_model`, and `codex_timeout_s`.

## Runtime layout

When an agent runs, BeaverFlow writes runtime state under the shared workspace:

```text
<workspace>/
  .agents/<agent-id>/...
  .workflow/shared/<channel>/...
```

These runtime directories are intentionally ignored by Git.

## Requirements

- Python 3.10+
- `openai` Python package
- `codex` CLI available in `PATH`
- the environment needed by your own task

## Quick start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the minimal sample in dry-run mode:

```bash
python3 agent_workflow_sample.py
```

It prints the commands that would be executed. To run it for real, set `DRY_RUN = False` in `agent_workflow_sample.py` and make sure your OpenAI and Codex environment is ready.

## Workflow DSL

Supported workflow expressions:

```text
A->B
A and B
loop(B, 20)
loop(A->B, 3)
A->loop(B, 20)
```

Semantics:

- `A->B`: run `A`, then `B`, on the same channel
- `A and B`: run both branches and write a branch manifest
- `loop(X, n)`: run `X` exactly `n` times

There is no retry primitive in the current design.

## Sample files

### `agent_workflow_sample.py`

A small writer/checker demo. It bootstraps one toy spec file under `demo_ws` and shows the basic loop shape.

### `agent_workflow_sample2.py`

A task-only template for a two-agent idea-generation / idea-testing workflow:

- Agent A appends new ideas to `idea_list`
- Agent B tests exactly one not-yet-tested idea per run

This sample intentionally does **not** contain counters, retry logic, or bootstrap helpers. Agent behavior is constrained only through `task` and `acceptance`.

Before a real run, edit the workspace and remote path constants in `agent_workflow_sample2.py` and prepare the workspace files it expects.

## Notes about the current interface

At the workflow layer, `Agent.task`, `Agent.acceptance`, and `Agent.mindset` are rendered as text templates. The workflow writes them into `.workflow/rendered/...` and then passes them to `agent.py` via `@file` arguments.

At the CLI layer, `agent.py` itself supports both inline text and `@path` for `--task`, `--acceptance`, and `--mindset`.

## Publishing and safety

Before making a public repository:

- do not commit `.agents/`, `.workflow/`, or workspace output directories
- do not commit SSH keys, host files, or remote access notes
- do not commit API keys, tokens, or `.env` files
- keep environment-specific paths inside local workspaces, not tracked source files

## License

This repository is released under the MIT License.
