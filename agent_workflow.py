#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"\s*(->|,|\(|\)|\d+|[A-Za-z_]\w*)")
KEYWORDS = {"and", "loop", "retry"}

DEFAULT_AI_MODEL = "gpt-5.4"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
DEFAULT_CODEX_TIMEOUT_S = 1800
DEFAULT_MAX_ROUNDS = 3


class WorkflowSyntaxError(SyntaxError):
    pass


@dataclass(slots=True)
class Agent:
    """Required: name, script, workspace, task, acceptance. mindset is optional; omitted means use agent.py builtin default."""
    name: str
    script: str | Path
    workspace: str | Path
    task: str
    acceptance: str
    mindset: str | None = None
    max_rounds: int = DEFAULT_MAX_ROUNDS
    reset: bool = False
    dry_run: bool = False
    ai_model: str | None = DEFAULT_AI_MODEL
    codex_model: str | None = DEFAULT_CODEX_MODEL
    codex_timeout_s: int | None = DEFAULT_CODEX_TIMEOUT_S
    base_dir: str | Path | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("name is required")
        if not str(self.script).strip():
            raise ValueError("script is required")
        if not str(self.workspace).strip():
            raise ValueError("workspace is required")

        self.task = str(self.task).strip()
        self.acceptance = str(self.acceptance).strip()
        self.mindset = None if self.mindset is None else str(self.mindset).strip() or None
        if not self.task:
            raise ValueError("task is required")
        if not self.acceptance:
            raise ValueError("acceptance is required")

        self.max_rounds = int(self.max_rounds)
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")

        self.reset = bool(self.reset)
        self.dry_run = bool(self.dry_run)

        self.ai_model = default_text(self.ai_model, DEFAULT_AI_MODEL)
        self.codex_model = default_text(self.codex_model, DEFAULT_CODEX_MODEL)
        self.codex_timeout_s = int(
            DEFAULT_CODEX_TIMEOUT_S if self.codex_timeout_s is None else self.codex_timeout_s
        )
        if self.codex_timeout_s <= 0:
            raise ValueError("codex_timeout_s must be positive")

        base_dir = Path(self.base_dir).expanduser() if self.base_dir else Path(__file__).resolve().parent
        self.base_dir = base_dir.resolve()

        script = Path(self.script).expanduser()
        if not script.is_absolute():
            script = (self.base_dir / script).resolve()
        self.script = script

        self.workspace = Path(self.workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def render(self, channel: str) -> tuple[str, str, str | None]:
        shared = channel_dir(self.workspace, channel)
        manifest = workflow_dir(self.workspace) / "branches" / f"{channel}.json"
        agent_id = f"wf.{channel}.{self.name}"
        variables = {
            "workspace": str(self.workspace),
            "shared": str(shared),
            "branch_manifest": str(manifest),
            "agent": self.name,
            "channel": channel,
            "agent_id": agent_id,
        }
        mindset = self.mindset.format(**variables) if self.mindset is not None else None
        return self.task.format(**variables), self.acceptance.format(**variables), mindset

    def materialize(self, channel: str) -> tuple[Path, Path, Path | None]:
        task, acceptance, mindset = self.render(channel)
        rendered = workflow_dir(self.workspace) / "rendered"
        rendered.mkdir(parents=True, exist_ok=True)
        stem = f"{channel}.{self.name}"
        task_path = rendered / f"{stem}.task.md"
        acceptance_path = rendered / f"{stem}.acceptance.md"
        task_path.write_text(task, encoding="utf-8")
        acceptance_path.write_text(acceptance, encoding="utf-8")
        mindset_path = None
        if mindset is not None:
            mindset_path = rendered / f"{stem}.mindset.md"
            mindset_path.write_text(mindset, encoding="utf-8")
        return task_path, acceptance_path, mindset_path

    def command(self, channel: str, *, reset: bool) -> list[str]:
        task_path, acceptance_path, mindset_path = self.materialize(channel)
        cmd = [
            sys.executable,
            str(self.script),
            "--workspace",
            str(self.workspace),
            "--agent-id",
            f"wf.{channel}.{self.name}",
            "--task",
            f"@{task_path}",
            "--acceptance",
            f"@{acceptance_path}",
            "--max-rounds",
            str(self.max_rounds),
        ]
        if mindset_path is not None:
            cmd += ["--mindset", f"@{mindset_path}"]
        if self.ai_model != DEFAULT_AI_MODEL:
            cmd += ["--ai-model", self.ai_model]
        if self.codex_model != DEFAULT_CODEX_MODEL:
            cmd += ["--codex-model", self.codex_model]
        if self.codex_timeout_s != DEFAULT_CODEX_TIMEOUT_S:
            cmd += ["--codex-timeout-s", str(self.codex_timeout_s)]
        if reset:
            cmd.append("--reset")
        return cmd

    def run(self, channel: str, *, first_run: bool) -> dict[str, Any]:
        cmd = self.command(channel, reset=self.reset and first_run)
        print("RUN", " ".join(cmd))
        if self.dry_run:
            return {"returncode": 0, "decision": "accepted", "summary": "dry-run", "status": {}}
        rc = subprocess.run(cmd, cwd=str(self.workspace)).returncode
        status = read_status(self.workspace, f"wf.{channel}.{self.name}")
        return {
            "returncode": rc,
            "decision": status.get("decision"),
            "summary": status.get("summary"),
            "status": status,
        }


def default_text(value: str | None, fallback: str) -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback


def workflow_dir(workspace: Path) -> Path:
    path = workspace / ".workflow"
    path.mkdir(parents=True, exist_ok=True)
    return path


def channel_dir(workspace: Path, channel: str) -> Path:
    path = workflow_dir(workspace) / "shared" / channel
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_status(workspace: Path, agent_id: str) -> dict[str, Any]:
    path = workspace / ".agents" / agent_id / "state" / "ACCEPTANCE_STATUS.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}




def write_branch_manifest(workspace: Path, channel: str, left: str, right: str) -> None:
    manifest = workflow_dir(workspace) / "branches" / f"{channel}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "channel": channel,
                "left_channel": left,
                "right_channel": right,
                "left_shared": str(channel_dir(workspace, left)),
                "right_shared": str(channel_dir(workspace, right)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        match = TOKEN_RE.match(text, pos)
        if not match:
            if text[pos].isspace():
                pos += 1
                continue
            raise WorkflowSyntaxError(f"unexpected character at position {pos}: {text[pos]!r}")
        tokens.append(match.group(1))
        pos = match.end()
    return tokens


def parse(text: str) -> Any:
    tokens = tokenize(text)
    if not tokens:
        raise WorkflowSyntaxError("empty workflow expression")
    index = 0

    def peek() -> str | None:
        return tokens[index] if index < len(tokens) else None

    def take(expected: str | None = None) -> str:
        nonlocal index
        token = peek()
        if token is None:
            raise WorkflowSyntaxError(f"unexpected end of expression; expected {expected or 'token'}")
        if expected is not None and token.lower() != expected.lower():
            raise WorkflowSyntaxError(f"expected {expected}, got {token}")
        index += 1
        return token

    def take_int() -> int:
        token = take()
        if not token.isdigit():
            raise WorkflowSyntaxError(f"expected integer, got {token}")
        return int(token)

    def term() -> Any:
        token = peek()
        if token is None:
            raise WorkflowSyntaxError("unexpected end of expression")
        low = token.lower()
        if low == "loop":
            take("loop")
            take("(")
            body = chain()
            take(",")
            rounds = take_int()
            take(")")
            return ("loop", body, rounds)
        if low == "retry":
            raise WorkflowSyntaxError("retry was removed; use loop(<expr>, <rounds>) for fixed repetition")
        if token == "(":
            take("(")
            node = chain()
            take(")")
            return node
        name = take()
        if name.lower() in KEYWORDS:
            raise WorkflowSyntaxError(f"reserved keyword cannot be used as agent name: {name}")
        return ("agent", name)

    def parallel() -> Any:
        node = term()
        while (peek() or "").lower() == "and":
            take("and")
            node = ("and", node, term())
        return node

    def chain() -> Any:
        node = parallel()
        while peek() == "->":
            take("->")
            node = ("seq", node, parallel())
        return node

    node = chain()
    if index != len(tokens):
        raise WorkflowSyntaxError(f"unexpected token: {peek()}")
    return node


def build_agent_lookup(agents: list[Agent] | tuple[Agent, ...]) -> dict[str, Agent]:
    if not agents:
        raise ValueError("agents cannot be empty")
    lookup: dict[str, Agent] = {}
    for index, agent in enumerate(agents):
        if not isinstance(agent, Agent):
            raise TypeError(f"agents[{index}] must be an Agent")
        key = agent.name.lower()
        if key in lookup:
            raise ValueError(f"duplicate agent name: {agent.name}")
        lookup[key] = agent
    return lookup


def resolve_workspace(agents: list[Agent] | tuple[Agent, ...]) -> Path:
    workspace = agents[0].workspace
    for index, agent in enumerate(agents[1:], start=1):
        if agent.workspace != workspace:
            raise ValueError(
                f"agents[0] and agents[{index}] use different workspaces: {workspace} != {agent.workspace}"
            )
    return workspace


def run_workflow(workflow: str, agents: list[Agent] | tuple[Agent, ...]) -> None:
    workspace = resolve_workspace(agents)
    lookup = build_agent_lookup(agents)
    seen: set[str] = set()

    def go(node: Any, channel: str) -> None:
        kind = node[0]
        if kind == "agent":
            name = node[1]
            agent = lookup.get(name.lower())
            if agent is None:
                raise KeyError(f"agent not found: {name}")
            agent_id = f"wf.{channel}.{agent.name}"
            result = agent.run(channel, first_run=agent_id not in seen)
            seen.add(agent_id)
            if result["returncode"] != 0 or result["decision"] != "accepted":
                raise RuntimeError(
                    f"{agent.name} failed on channel={channel}: rc={result['returncode']}, "
                    f"decision={result['decision']}, summary={result['summary']}"
                )
            return
        if kind == "seq":
            go(node[1], channel)
            go(node[2], channel)
            return
        if kind == "and":
            left, right = f"{channel}.left", f"{channel}.right"
            go(node[1], left)
            go(node[2], right)
            write_branch_manifest(workspace, channel, left, right)
            return
        if kind == "loop":
            body, rounds = node[1], int(node[2])
            if rounds <= 0:
                raise ValueError("loop rounds must be positive")
            for index in range(rounds):
                print(f"LOOP {index + 1}/{rounds} channel={channel}")
                go(body, channel)
            return
        raise ValueError(f"unknown workflow node: {kind}")

    go(parse(workflow), "main")
