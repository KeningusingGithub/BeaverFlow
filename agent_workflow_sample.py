#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from agent_workflow import Agent, channel_dir, run_workflow

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE / "demo_ws"
WORKFLOW = "loop(A->B, 3)"
DRY_RUN = True  # Set to False for a real run.


def bootstrap_workspace() -> None:
    shared = channel_dir(WORKSPACE, "main")
    spec = {
        "expected_text": "READY: BeaverFlow demo",
        "goal": "Create one exact text file and verify it with a mechanical workflow loop.",
    }
    (shared / "spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


A_TASK = (
    "Read {shared}/spec.json and optional {shared}/feedback.md. "
    "Write {shared}/answer.txt using exactly the expected_text value from spec.json."
)
A_ACCEPTANCE = (
    "- {shared}/answer.txt exists\n"
    "- {shared}/answer.txt is non-empty\n"
    "- {shared}/answer.txt exactly matches expected_text in {shared}/spec.json"
)

B_TASK = (
    "Read {shared}/spec.json and {shared}/answer.txt. "
    "Write {shared}/status.json as valid JSON with keys ok and reason. "
    "If ok is false, also write {shared}/feedback.md with one short repair instruction."
)
B_ACCEPTANCE = (
    "- {shared}/status.json exists and is valid JSON\n"
    "- JSON contains boolean field ok\n"
    "- JSON contains non-empty string field reason\n"
    "- When ok is false, {shared}/feedback.md exists and is non-empty"
)

A = Agent(
    name="A",
    script="./agent.py",
    workspace=WORKSPACE,
    task=A_TASK,
    acceptance=A_ACCEPTANCE,
    reset=True,
    dry_run=DRY_RUN,
)

B = Agent(
    name="B",
    script="./agent.py",
    workspace=WORKSPACE,
    task=B_TASK,
    acceptance=B_ACCEPTANCE,
    max_rounds=2,
    reset=True,
    dry_run=DRY_RUN,
)

AGENTS = [A, B]


if __name__ == "__main__":
    bootstrap_workspace()
    run_workflow(WORKFLOW, AGENTS)
    print(f"done: {channel_dir(WORKSPACE, 'main')}")
