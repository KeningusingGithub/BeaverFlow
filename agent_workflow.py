#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"\s*(->|,|\(|\)|\d+|[A-Za-z_]\w*)")
KEYWORDS = {"and", "loop", "retry"}

DEFAULT_AI_MODEL = "gpt-5.4"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
DEFAULT_CODEX_TIMEOUT_S = 1800
DEFAULT_MAX_ROUNDS = 3
MAX_AGENT_ID_LEN = 120


class WorkflowSyntaxError(SyntaxError):
    pass


# -------------------------
# Small helpers
# -------------------------

def default_text(value: str | None, fallback: str) -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback


def sanitize_component(raw: str) -> str:
    value = str(raw or "").strip()
    value = value.replace("\\", "-").replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip(" ._-")
    return value or "x"


def normalize_agent_id(raw: str | None) -> str:
    value = str(raw or "").strip()
    value = value.replace("\\", "-").replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip(" ._-")
    if not value:
        raise ValueError("agent-id is empty after normalization")
    return value[:MAX_AGENT_ID_LEN]


def workflow_dir(workspace: Path) -> Path:
    path = workspace / ".workflow"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_workflow_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


def normalize_workflow_run_id(raw: str | None) -> str:
    value = "" if raw is None else str(raw).strip()
    if not value:
        value = new_workflow_run_id()
    value = value.replace("\\", "-").replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip(" ._-")
    if not value:
        value = new_workflow_run_id()
    return value


def workflow_run_dir(workspace: Path, workflow_run_id: str) -> Path:
    path = workflow_dir(workspace) / "runs" / workflow_run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def rendered_dir(workspace: Path, workflow_run_id: str) -> Path:
    path = workflow_run_dir(workspace, workflow_run_id) / "rendered"
    path.mkdir(parents=True, exist_ok=True)
    return path


def channel_dir(workspace: Path, workflow_run_id: str, channel: str) -> Path:
    path = workflow_run_dir(workspace, workflow_run_id) / "shared" / channel
    path.mkdir(parents=True, exist_ok=True)
    return path


def branch_manifest_path(workspace: Path, workflow_run_id: str, channel: str) -> Path:
    path = workflow_run_dir(workspace, workflow_run_id) / "branches" / f"{channel}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def safe_file_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def build_execution_agent_id(
    workflow_run_id: str,
    channel: str,
    agent_name: str,
    invocation_index: int,
) -> str:
    """
    One workflow invocation -> one fresh agent_id.

    This is the core correctness boundary:
    - workflow controls *how many invocations* happen
    - agent.py max_rounds controls *how many internal rounds* happen inside one invocation
    - reset is no longer relied upon for correctness
    """
    workflow_run_id = sanitize_component(workflow_run_id)
    channel = sanitize_component(channel)
    agent_name = sanitize_component(agent_name)
    suffix = f"call_{invocation_index:04d}"
    raw = f"wf.{workflow_run_id}.{channel}.{agent_name}.{suffix}"
    if len(raw) <= MAX_AGENT_ID_LEN:
        return raw

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    shortened = f"wf.{workflow_run_id[:32]}.{digest}.{suffix}"
    return normalize_agent_id(shortened)


def read_status(workspace: Path, agent_id: str) -> dict[str, Any]:
    path = workspace / ".agents" / agent_id / "state" / "ACCEPTANCE_STATUS.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def write_branch_manifest(
    workspace: Path,
    workflow_run_id: str,
    channel: str,
    left: str,
    right: str,
) -> None:
    manifest = branch_manifest_path(workspace, workflow_run_id, channel)
    manifest.write_text(
        json.dumps(
            {
                "workflow_run_id": workflow_run_id,
                "channel": channel,
                "left_channel": left,
                "right_channel": right,
                "left_shared": str(channel_dir(workspace, workflow_run_id, left)),
                "right_shared": str(channel_dir(workspace, workflow_run_id, right)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# -------------------------
# Agent
# -------------------------

@dataclass(slots=True)
class Agent:
    """
    Required: name, script, workspace, task, acceptance.
    mindset is optional; omitted means use agent.py builtin default.

    Notes:
    - Each workflow invocation now gets a fresh agent_id.
    - `reset` is kept only for compatibility / manual debugging. It is no longer
      the mechanism that guarantees loop correctness.
    """

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

    def render(
        self,
        *,
        workflow_run_id: str,
        channel: str,
        agent_id: str,
    ) -> tuple[str, str, str | None]:
        shared = channel_dir(self.workspace, workflow_run_id, channel)
        manifest = branch_manifest_path(self.workspace, workflow_run_id, channel)
        variables = {
            "workspace": str(self.workspace),
            "workflow_root": str(workflow_dir(self.workspace)),
            "workflow_run_id": workflow_run_id,
            "workflow_run_dir": str(workflow_run_dir(self.workspace, workflow_run_id)),
            "shared": str(shared),
            "branch_manifest": str(manifest),
            "agent": self.name,
            "logical_agent": self.name,
            "channel": channel,
            "agent_id": agent_id,
        }
        mindset = self.mindset.format(**variables) if self.mindset is not None else None
        return self.task.format(**variables), self.acceptance.format(**variables), mindset

    def materialize(
        self,
        *,
        workflow_run_id: str,
        channel: str,
        agent_id: str,
    ) -> tuple[Path, Path, Path | None]:
        task, acceptance, mindset = self.render(
            workflow_run_id=workflow_run_id,
            channel=channel,
            agent_id=agent_id,
        )
        rendered = rendered_dir(self.workspace, workflow_run_id)
        stem = safe_file_stem(agent_id)

        task_path = rendered / f"{stem}.task.md"
        acceptance_path = rendered / f"{stem}.acceptance.md"
        task_path.write_text(task, encoding="utf-8")
        acceptance_path.write_text(acceptance, encoding="utf-8")

        mindset_path = None
        if mindset is not None:
            mindset_path = rendered / f"{stem}.mindset.md"
            mindset_path.write_text(mindset, encoding="utf-8")

        return task_path, acceptance_path, mindset_path

    def command(
        self,
        *,
        workflow_run_id: str,
        channel: str,
        agent_id: str,
    ) -> list[str]:
        task_path, acceptance_path, mindset_path = self.materialize(
            workflow_run_id=workflow_run_id,
            channel=channel,
            agent_id=agent_id,
        )
        cmd = [
            sys.executable,
            str(self.script),
            "--workspace",
            str(self.workspace),
            "--agent-id",
            agent_id,
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
        if self.reset:
            # Usually redundant because agent_id is already fresh every time.
            # Kept only for compatibility / manual debugging semantics.
            cmd.append("--reset")
        return cmd

    def run(
        self,
        *,
        workflow_run_id: str,
        channel: str,
        agent_id: str,
    ) -> dict[str, Any]:
        cmd = self.command(
            workflow_run_id=workflow_run_id,
            channel=channel,
            agent_id=agent_id,
        )
        print("RUN", " ".join(cmd))
        if self.dry_run:
            return {
                "returncode": 0,
                "decision": "accepted",
                "summary": "dry-run",
                "status": {},
                "agent_id": agent_id,
            }

        rc = subprocess.run(cmd, cwd=str(self.workspace)).returncode
        status = read_status(self.workspace, agent_id)
        return {
            "returncode": rc,
            "decision": status.get("decision"),
            "summary": status.get("summary"),
            "status": status,
            "agent_id": agent_id,
        }


# -------------------------
# Workflow parser
# -------------------------

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


# -------------------------
# Workflow runtime
# -------------------------

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


def run_workflow(
    workflow: str,
    agents: list[Agent] | tuple[Agent, ...],
    *,
    workflow_run_id: str | None = None,
) -> str:
    workspace = resolve_workspace(agents)
    lookup = build_agent_lookup(agents)

    workflow_run_id = normalize_workflow_run_id(workflow_run_id)
    invocation_counts: dict[tuple[str, str], int] = defaultdict(int)

    run_meta_path = workflow_run_dir(workspace, workflow_run_id) / "RUN_META.json"
    run_meta_path.write_text(
        json.dumps(
            {
                "workflow": workflow,
                "workflow_run_id": workflow_run_id,
                "workspace": str(workspace),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "agents": [agent.name for agent in agents],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def go(node: Any, channel: str) -> None:
        kind = node[0]

        if kind == "agent":
            name = node[1]
            agent = lookup.get(name.lower())
            if agent is None:
                raise KeyError(f"agent not found: {name}")

            key = (channel, agent.name)
            invocation_counts[key] += 1
            invocation_index = invocation_counts[key]

            agent_id = build_execution_agent_id(
                workflow_run_id=workflow_run_id,
                channel=channel,
                agent_name=agent.name,
                invocation_index=invocation_index,
            )

            result = agent.run(
                workflow_run_id=workflow_run_id,
                channel=channel,
                agent_id=agent_id,
            )
            if result["returncode"] != 0:
                raise RuntimeError(
                    f"{agent.name} execution failed on channel={channel}: "
                    f"rc={result['returncode']}, "
                    f"decision={result['decision']}, "
                    f"summary={result['summary']}, "
                    f"agent_id={agent_id}"
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
            write_branch_manifest(workspace, workflow_run_id, channel, left, right)
            return

        if kind == "loop":
            body, rounds = node[1], int(node[2])
            if rounds <= 0:
                raise ValueError("loop rounds must be positive")
            for index in range(rounds):
                print(
                    f"LOOP {index + 1}/{rounds} "
                    f"channel={channel} workflow_run_id={workflow_run_id}"
                )
                go(body, channel)
            return

        raise ValueError(f"unknown workflow node: {kind}")

    go(parse(workflow), "main")
    return workflow_run_id
