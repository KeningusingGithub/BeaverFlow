#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal generic AI <-> Codex loop with agent namespaces.

Design goals:
- single-file script
- minimal public interface: workspace + task + acceptance
- task and acceptance are separate truths, not merged
- task/acceptance are stored as files and injected into prompts at runtime
- multiple agents can share one workspace without clobbering each other

Execution model:
- Codex runs with cwd=<workspace>, so it can inspect and modify task-relevant
  project files anywhere inside the shared workspace.
- Agent bookkeeping files (spec/state/runs/snapshots/status) are stored under
  <workspace>/.agents/<agent-id>/ so multiple agents do not collide.
- This is namespace isolation for bookkeeping, not hard permission isolation.

Namespace layout:
  <workspace>/.agents/<agent-id>/
    spec/
      task.md
      acceptance.md
      mindset.md
    state/
      PROGRESS.md
      HUMAN.md
      TASK.snapshot.md
      ACCEPTANCE.snapshot.md
      ACCEPTANCE_ITEMS.json
      ACCEPTANCE_STATUS.json
      RUN_META.json
      runs/

Important behavior:
- `--agent-id` isolates one agent from others inside the same workspace.
- `--reset` clears only this agent's `state/` directory; it preserves `spec/`.
- if `--task` / `--acceptance` are provided, their contents are mirrored into this
  agent's `spec/` directory first; if omitted, the agent reads the default files
  from its own `spec/`.
- if `--mindset` is provided, its content overrides the default mindset for this
  run and is mirrored into `spec/mindset.md`.
- if `--mindset` is omitted, the built-in default engineering mindset is used and
  mirrored into `spec/mindset.md` for this run.

Typical usage:
  python agent.py --workspace . --agent-id reviewer-a
  python agent.py --workspace . --agent-id reviewer-a --task @task.md --acceptance @acceptance.md
  python agent.py --workspace . --agent-id reviewer-a --task "do X" --acceptance "1. Y passes 2. Z exists"
  python agent.py --workspace . --agent-id reviewer-a --ai-model gpt-5.4 --codex-model gpt-5.3-codex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_AI_MODEL = "gpt-5.4"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
DEFAULT_CODEX_TIMEOUT_S = 1800
DEFAULT_MAX_ROUNDS = 5
DEFAULT_LOG_TAIL_BYTES = 12_000
DEFAULT_CONTEXT_TAIL_CHARS = 3500
DEFAULT_MINDSET_MAX_CHARS = 8000

AGENTS_DIRNAME = ".agents"
SPEC_DIRNAME = "spec"
STATE_DIRNAME = "state"
RUNS_DIRNAME = "runs"

DEFAULT_AGENT_ID = "default"
DEFAULT_TASK_SOURCE_FILENAME = "task.md"
DEFAULT_ACCEPTANCE_SOURCE_FILENAME = "acceptance.md"
DEFAULT_MINDSET_SOURCE_FILENAME = "mindset.md"

PROGRESS_FILENAME = "PROGRESS.md"
HUMAN_FILENAME = "HUMAN.md"
TASK_SNAPSHOT_FILENAME = "TASK.snapshot.md"
ACCEPTANCE_SNAPSHOT_FILENAME = "ACCEPTANCE.snapshot.md"
ACCEPTANCE_ITEMS_FILENAME = "ACCEPTANCE_ITEMS.json"
ACCEPTANCE_STATUS_FILENAME = "ACCEPTANCE_STATUS.json"
RUN_META_FILENAME = "RUN_META.json"


DEFAULT_MINDSET_TEXT = """# MINDSET
> Default cross-task engineering principles. Override with `--mindset` only when a specific agent truly needs it.

- Before each round, align on the task goal and acceptance requirements before taking action.
- Prefer the smallest verifiable change; do not spread too wide in a single round.
- Prioritize facts and evidence; do not assume the environment state without evidence.
- Update your own `state/PROGRESS.md` every round, clearly stating what changed, what was verified, and what comes next.
- For useless temporary files, debug output, caches, or intermediate artifacts you created yourself, clean them up promptly whenever possible.
- Truly important completion conditions must be written in acceptance, not hidden in mindset.
- Without item-by-item acceptance evidence, you cannot claim completion.
- To complete the task, you may read and write task-relevant project files across the entire workspace; however, the agent's own bookkeeping may only be written inside its own namespace, and you must not touch other agents' namespaces.
- If the task cannot be completed, you may end honestly, but you must clearly state which acceptance items failed.
"""


AI_INSTRUCTIONS = """You are the AI (Planner), operating inside an alternating AI <-> Codex loop.

You will see the following in the context:
- TASK SNAPSHOT: the task objective, which explains what must be done
- ACCEPTANCE SNAPSHOT: the acceptance target, which explains what counts as complete
- ACCEPTANCE STATUS: the current working status of each acceptance item
- OUTSTANDING / FAILED ACCEPTANCE ITEMS: the acceptance items that are still unmet
- MINDSET: cross-task execution principles
- AGENT NAMESPACE: information about your own agent namespace

Key boundaries:
- task decides what must be done.
- acceptance decides when the task can end.
- mindset contains only general discipline; it does not replace acceptance.
- To complete the task, you may read and write project files across the entire workspace; however, the agent's own spec/state/runs may only be maintained inside its own namespace, and you must not touch other agents' namespaces.
- `spec/task.md` / `spec/acceptance.md` are the source inputs for the current agent; do not treat editing these source files as a way to complete the task.
- Ordinary project files in the workspace (for example src/, tests/, README.md) may be modified normally if the task requires it.

Real-world constraints:
- You cannot directly inspect workspace files yourself; you can only perceive the environment through Codex execution results, logs, and files written to disk.
- Codex is your eyes and hands: it can read files, write files, and run commands.
- The orchestrator executes your instructions via `codex exec`, with approvals disabled.

Your goals:
- Prefer the smallest next step that is verifiable.
- Every round should focus on which acceptance items are still missing, rather than doing vague work.
- Important acceptance items must not rely only on memory; maintain working memory continuously through `ACCEPTANCE_STATUS.json`.
- You may output `ACTION: FINISH` only when the acceptance gate truly passes.
- If the task cannot be completed, you may still end honestly, but `ACCEPTANCE_STATUS.json` must explain the failure evidence item by item and satisfy the gate for `rejected`.

Output format (must be followed):
- The first line must be exactly: ACTION: RUN_CODEX or ACTION: ASK_HUMAN or ACTION: FINISH

If ACTION: RUN_CODEX, use:
BEGIN_CODEX
...Natural-language instructions for Codex. They must be specific, executable, and verifiable...
END_CODEX

If ACTION: ASK_HUMAN, use:
BEGIN_QUESTION
...Question for the human...
END_QUESTION

If ACTION: FINISH, use:
BEGIN_FINAL
...Final summary: completion status, artifact locations, limitations, and next steps...
END_FINAL
"""


ACTION_RE = re.compile(r"^\s*ACTION\s*:\s*(RUN_CODEX|ASK_HUMAN|FINISH)\s*$", re.IGNORECASE | re.MULTILINE)
LIST_ITEM_RE = re.compile(
    "^\\s*(?:[-*+]|\\d+[\\.)]|[\\uFF08(]?\\d+[)\\uFF09]|[A-Za-z][\\.)]|[\\u2460-\\u2469]|[\\u4e00\\u4e8c\\u4e09\\u56db\\u4e94\\u516d\\u4e03\\u516b\\u4e5d\\u5341]+[\\u3001.\\uFF0E])\\s+(.*\\S)\\s*$"
)


@dataclass(frozen=True)
class AgentPaths:
    workspace: Path
    agent_id: str
    agents_root: Path
    agent_home: Path
    spec_dir: Path
    state_dir: Path
    runs_dir: Path
    task_source_path: Path
    acceptance_source_path: Path
    mindset_source_path: Path
    progress_path: Path
    human_path: Path
    task_snapshot_path: Path
    acceptance_snapshot_path: Path
    acceptance_items_path: Path
    acceptance_status_path: Path
    run_meta_path: Path


# -------------------------
# Paths / layout
# -------------------------

def normalize_agent_id(raw: Optional[str]) -> str:
    value = str(raw or "").strip() or DEFAULT_AGENT_ID
    value = value.replace("\\", "-").replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip(" ._-")
    if not value:
        raise ValueError("agent-id is empty after normalization")
    return value[:120]


def build_paths(workspace: Path, agent_id: str) -> AgentPaths:
    aid = normalize_agent_id(agent_id)
    agents_root = workspace / AGENTS_DIRNAME
    agent_home = agents_root / aid
    spec_dir = agent_home / SPEC_DIRNAME
    state_dir = agent_home / STATE_DIRNAME
    return AgentPaths(
        workspace=workspace,
        agent_id=aid,
        agents_root=agents_root,
        agent_home=agent_home,
        spec_dir=spec_dir,
        state_dir=state_dir,
        runs_dir=state_dir / RUNS_DIRNAME,
        task_source_path=spec_dir / DEFAULT_TASK_SOURCE_FILENAME,
        acceptance_source_path=spec_dir / DEFAULT_ACCEPTANCE_SOURCE_FILENAME,
        mindset_source_path=spec_dir / DEFAULT_MINDSET_SOURCE_FILENAME,
        progress_path=state_dir / PROGRESS_FILENAME,
        human_path=state_dir / HUMAN_FILENAME,
        task_snapshot_path=state_dir / TASK_SNAPSHOT_FILENAME,
        acceptance_snapshot_path=state_dir / ACCEPTANCE_SNAPSHOT_FILENAME,
        acceptance_items_path=state_dir / ACCEPTANCE_ITEMS_FILENAME,
        acceptance_status_path=state_dir / ACCEPTANCE_STATUS_FILENAME,
        run_meta_path=state_dir / RUN_META_FILENAME,
    )


def ensure_layout(paths: AgentPaths) -> None:
    paths.spec_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)

    if not paths.progress_path.exists():
        paths.progress_path.write_text(
            "# PROGRESS\n\n"
            f"- agent_id: {paths.agent_id}\n"
            f"- namespace: {paths.agent_home}\n"
            "- This file is maintained by the agent.\n"
            "- Each round should add one short entry: what changed, what was verified, what remains.\n",
            encoding="utf-8",
        )

    if not paths.human_path.exists():
        paths.human_path.write_text(
            "# HUMAN INPUTS\n\n"
            f"- agent_id: {paths.agent_id}\n"
            "- Questions asked by AI and the human responses.\n",
            encoding="utf-8",
        )

    if not paths.mindset_source_path.exists():
        paths.mindset_source_path.write_text(DEFAULT_MINDSET_TEXT, encoding="utf-8")


def reset_agent_state(paths: AgentPaths) -> None:
    if paths.state_dir.exists():
        shutil.rmtree(paths.state_dir, ignore_errors=True)


# -------------------------
# Small helpers
# -------------------------

def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def append_progress(paths: AgentPaths, line: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with paths.progress_path.open("a", encoding="utf-8") as f:
        f.write(f"- {ts} {line}\n")


def append_human(paths: AgentPaths, question: str, answer: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with paths.human_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {ts}\n**Q:** {question}\n\n**A:** {answer}\n")


def read_text_limited(path: Path, max_chars: int, *, tail: bool = False) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="replace")
    if max_chars <= 0 or len(txt) <= max_chars:
        return txt
    if tail:
        return txt[-max_chars:]
    return txt[:max_chars] + "\n\n...[TRUNCATED]\n"


def one_line(s: str, max_len: int = 180) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 12] + " ...[cut]"


def resolve_input_path(workspace: Path, raw_path: str) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p
    return (workspace / p).resolve()


def normalize_multiline_text(text: str) -> str:
    lines = [line.rstrip() for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip() + "\n"


def compute_spec_hash(task_text: str, acceptance_text: str) -> str:
    payload = json.dumps(
        {
            "task": normalize_multiline_text(task_text),
            "acceptance": normalize_multiline_text(acceptance_text),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_text_source(*, workspace: Path, raw: Optional[str], default_path: Optional[Path], label: str, required: bool) -> Tuple[str, Dict[str, str]]:
    if raw is None:
        if default_path is not None and default_path.exists():
            if not default_path.is_file():
                raise ValueError(f"default {label} path is not a regular file: {default_path}")
            return (
                default_path.read_text(encoding="utf-8", errors="replace").strip(),
                {"mode": "default_file", "ref": str(default_path)},
            )
        if required:
            missing = str(default_path) if default_path is not None else f"--{label}"
            raise ValueError(f"missing {label}; provide --{label} or create {missing}")
        return "", {"mode": "empty", "ref": ""}

    value = str(raw)
    if value.startswith("@"):
        path = resolve_input_path(workspace, value[1:])
        if not path.exists() or not path.is_file():
            raise ValueError(f"{label} file not found: {path}")
        return (
            path.read_text(encoding="utf-8", errors="replace").strip(),
            {"mode": "explicit_file", "ref": str(path)},
        )

    return value.strip(), {"mode": "inline", "ref": "inline"}


def load_mindset_source(*, workspace: Path, raw: Optional[str]) -> Tuple[str, Dict[str, str]]:
    if raw is None:
        return DEFAULT_MINDSET_TEXT.strip(), {"mode": "builtin_default", "ref": "DEFAULT_MINDSET_TEXT"}
    value = str(raw)
    if value.startswith("@"):
        path = resolve_input_path(workspace, value[1:])
        if not path.exists() or not path.is_file():
            raise ValueError(f"mindset file not found: {path}")
        return (
            path.read_text(encoding="utf-8", errors="replace").strip(),
            {"mode": "explicit_file", "ref": str(path)},
        )
    return value.strip(), {"mode": "inline", "ref": "inline"}


def load_required_source_files(paths: AgentPaths) -> Tuple[str, str]:
    if not paths.task_source_path.exists():
        raise ValueError(f"missing task source file: {paths.task_source_path}")
    if not paths.acceptance_source_path.exists():
        raise ValueError(f"missing acceptance source file: {paths.acceptance_source_path}")
    task_text = paths.task_source_path.read_text(encoding="utf-8", errors="replace").strip()
    acceptance_text = paths.acceptance_source_path.read_text(encoding="utf-8", errors="replace").strip()
    if not task_text:
        raise ValueError(f"task source is empty: {paths.task_source_path}")
    if not acceptance_text:
        raise ValueError(f"acceptance source is empty: {paths.acceptance_source_path}")
    return task_text, acceptance_text


def initialize_namespace_sources(
    *,
    paths: AgentPaths,
    workspace: Path,
    task_raw: Optional[str],
    acceptance_raw: Optional[str],
    mindset_raw: Optional[str],
) -> Dict[str, Dict[str, str]]:
    task_text, task_info = load_text_source(
        workspace=workspace,
        raw=task_raw,
        default_path=paths.task_source_path,
        label="task",
        required=True,
    )
    acceptance_text, acceptance_info = load_text_source(
        workspace=workspace,
        raw=acceptance_raw,
        default_path=paths.acceptance_source_path,
        label="acceptance",
        required=True,
    )
    mindset_text, mindset_info = load_mindset_source(workspace=workspace, raw=mindset_raw)

    paths.task_source_path.write_text(normalize_multiline_text(task_text), encoding="utf-8")
    paths.acceptance_source_path.write_text(normalize_multiline_text(acceptance_text), encoding="utf-8")
    paths.mindset_source_path.write_text(normalize_multiline_text(mindset_text), encoding="utf-8")

    return {
        "task": task_info,
        "acceptance": acceptance_info,
        "mindset": mindset_info,
    }


def write_run_meta(paths: AgentPaths, source_meta: Dict[str, Dict[str, str]]) -> None:
    task_text, acceptance_text = load_required_source_files(paths)
    spec_hash = compute_spec_hash(task_text, acceptance_text)
    write_json(
        paths.run_meta_path,
        {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent_id": paths.agent_id,
            "workspace": str(paths.workspace),
            "agent_home": str(paths.agent_home),
            "spec_dir": str(paths.spec_dir),
            "state_dir": str(paths.state_dir),
            "paths": {
                "task_source": str(paths.task_source_path),
                "acceptance_source": str(paths.acceptance_source_path),
                "mindset_source": str(paths.mindset_source_path),
                "progress": str(paths.progress_path),
                "human": str(paths.human_path),
                "task_snapshot": str(paths.task_snapshot_path),
                "acceptance_snapshot": str(paths.acceptance_snapshot_path),
                "acceptance_items": str(paths.acceptance_items_path),
                "acceptance_status": str(paths.acceptance_status_path),
                "runs_dir": str(paths.runs_dir),
            },
            "input_sources": source_meta,
            "spec_hash_at_start": spec_hash,
            "notes": [
                "spec/task.md and spec/acceptance.md are the canonical files for this agent namespace.",
                "Codex runs with cwd=<workspace>, so task-relevant project files may live anywhere in the shared workspace.",
                "state/ is disposable runtime state; --reset clears only state/.",
            ],
        },
    )


# -------------------------
# Acceptance source -> items -> status
# -------------------------

def looks_like_acceptance_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    low = stripped.lower()
    if low in {"begin_acceptance", "end_acceptance"}:
        return True
    if stripped.startswith("#"):
        core = stripped.lstrip("#").strip().lower()
        core = re.sub(r"\s+", " ", core)
        if core in {"acceptance", "acceptance requirements", "\u9a8c\u6536", "\u9a8c\u6536\u8981\u6c42"}:
            return True
    return False


def parse_acceptance_items(text: str) -> List[Dict[str, str]]:
    text = normalize_multiline_text(text).strip()
    if not text:
        raise ValueError("acceptance is empty")

    items: List[str] = []
    current: Optional[str] = None
    saw_list_item = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if looks_like_acceptance_heading(line):
            continue

        m = LIST_ITEM_RE.match(line)
        if m:
            if current:
                items.append(current.strip())
            current = m.group(1).strip()
            saw_list_item = True
            continue

        stripped = line.strip()
        if not stripped:
            if saw_list_item and current:
                items.append(current.strip())
                current = None
            continue

        if saw_list_item:
            if current:
                current += " " + stripped
            else:
                current = stripped

    if saw_list_item:
        if current:
            items.append(current.strip())
    else:
        paragraphs = []
        for part in re.split(r"\n\s*\n+", text):
            candidate = part.strip()
            if not candidate:
                continue
            if looks_like_acceptance_heading(candidate):
                continue
            paragraphs.append(candidate)
        if paragraphs:
            items = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
        else:
            flattened = re.sub(r"\s+", " ", text).strip()
            if looks_like_acceptance_heading(flattened):
                flattened = ""
            items = [flattened] if flattened else []

    cleaned: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append({"id": f"A{len(cleaned) + 1}", "text": normalized})

    if not cleaned:
        raise ValueError("could not parse any acceptance items")
    return cleaned


def build_empty_acceptance_status(spec_hash: str, items: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "spec_hash": spec_hash,
        "decision": "incomplete",
        "summary": "",
        "requirements": [
            {"id": item["id"], "text": item["text"], "status": "unknown", "evidence": ""}
            for item in items
        ],
        "artifacts": [],
        "notes": "",
        "next_steps": [],
    }


def normalize_existing_status(existing: Dict[str, Any], spec_hash: str, items: List[Dict[str, str]]) -> Dict[str, Any]:
    by_id: Dict[str, Dict[str, Any]] = {}
    reqs = existing.get("requirements")
    if isinstance(reqs, list):
        for req in reqs:
            if isinstance(req, dict):
                rid = str(req.get("id", "") or "").strip()
                if rid:
                    by_id[rid] = req

    normalized = build_empty_acceptance_status(spec_hash, items)
    normalized["decision"] = str(existing.get("decision", "incomplete") or "incomplete").strip() or "incomplete"
    normalized["summary"] = str(existing.get("summary", "") or "")
    normalized["notes"] = str(existing.get("notes", "") or "")

    next_steps = existing.get("next_steps")
    if isinstance(next_steps, list):
        normalized["next_steps"] = [str(x) for x in next_steps if str(x).strip()]

    artifacts = existing.get("artifacts")
    if isinstance(artifacts, list):
        normalized["artifacts"] = [str(x) for x in artifacts if str(x).strip()]

    new_reqs: List[Dict[str, Any]] = []
    for item in items:
        old = by_id.get(item["id"], {})
        req = {
            "id": item["id"],
            "text": item["text"],
            "status": str(old.get("status", "unknown") or "unknown").strip().lower() or "unknown",
            "evidence": str(old.get("evidence", "") or ""),
        }
        new_reqs.append(req)
    normalized["requirements"] = new_reqs
    return normalized


def sync_spec_files(paths: AgentPaths) -> Tuple[str, List[Dict[str, str]]]:
    task_text, acceptance_text = load_required_source_files(paths)
    spec_hash = compute_spec_hash(task_text, acceptance_text)
    items = parse_acceptance_items(acceptance_text)

    paths.task_snapshot_path.write_text(normalize_multiline_text(task_text), encoding="utf-8")
    paths.acceptance_snapshot_path.write_text(normalize_multiline_text(acceptance_text), encoding="utf-8")
    write_json(paths.acceptance_items_path, {"spec_hash": spec_hash, "items": items})

    existing: Optional[Dict[str, Any]] = None
    if paths.acceptance_status_path.exists() and paths.acceptance_status_path.is_file():
        try:
            candidate = read_json(paths.acceptance_status_path)
            if isinstance(candidate, dict) and str(candidate.get("spec_hash", "")) == spec_hash:
                existing = candidate
        except Exception:
            existing = None

    status = normalize_existing_status(existing or {}, spec_hash, items)
    if existing is None:
        status = build_empty_acceptance_status(spec_hash, items)
    write_json(paths.acceptance_status_path, status)

    return spec_hash, items


def resolve_artifact_path(paths: AgentPaths, raw: Any) -> Optional[Path]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = (paths.workspace / p).resolve()
    return p


def load_acceptance_items(paths: AgentPaths) -> Tuple[Optional[Dict[str, Any]], str]:
    p = paths.acceptance_items_path
    if not p.exists():
        return None, f"missing {p}"
    if not p.is_file():
        return None, f"{p} is not a regular file"
    try:
        data = read_json(p)
    except Exception as e:
        return None, f"invalid ACCEPTANCE_ITEMS.json: {e}"
    if not isinstance(data, dict):
        return None, "ACCEPTANCE_ITEMS.json root must be an object"
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return None, "ACCEPTANCE_ITEMS.json must contain a non-empty items list"
    return data, ""


def load_acceptance_status(paths: AgentPaths) -> Tuple[Optional[Dict[str, Any]], str]:
    p = paths.acceptance_status_path
    if not p.exists():
        return None, f"missing {p}"
    if not p.is_file():
        return None, f"{p} is not a regular file"
    try:
        data = read_json(p)
    except Exception as e:
        return None, f"invalid ACCEPTANCE_STATUS.json: {e}"
    if not isinstance(data, dict):
        return None, "ACCEPTANCE_STATUS.json root must be an object"
    return data, ""


def evaluate_acceptance_gate(paths: AgentPaths) -> Dict[str, Any]:
    items_doc, items_err = load_acceptance_items(paths)
    status_doc, status_err = load_acceptance_status(paths)

    result: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "summary": "",
        "decision": None,
        "counts": {"pass": 0, "fail": 0, "unknown": 0},
        "items": [],
        "data": status_doc,
    }

    if items_doc is None:
        result["reason"] = items_err
        result["summary"] = items_err
        return result
    if status_doc is None:
        result["reason"] = status_err
        result["summary"] = status_err
        return result

    spec_hash_items = str(items_doc.get("spec_hash", "") or "").strip()
    spec_hash_status = str(status_doc.get("spec_hash", "") or "").strip()
    if not spec_hash_items or spec_hash_status != spec_hash_items:
        result["reason"] = "acceptance status does not match the current task/acceptance snapshot"
        result["summary"] = result["reason"]
        return result

    items_raw = items_doc.get("items") or []
    canonical: Dict[str, Dict[str, str]] = {}
    for idx, item in enumerate(items_raw):
        if not isinstance(item, dict):
            result["reason"] = f"items[{idx}] must be an object"
            result["summary"] = result["reason"]
            return result
        rid = str(item.get("id", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        if not rid or not text:
            result["reason"] = f"items[{idx}] must contain id and text"
            result["summary"] = result["reason"]
            return result
        if rid in canonical:
            result["reason"] = f"duplicate requirement id: {rid}"
            result["summary"] = result["reason"]
            return result
        canonical[rid] = {"id": rid, "text": text}

    decision = str(status_doc.get("decision", "") or "").strip().lower()
    result["decision"] = decision
    result["summary"] = str(status_doc.get("summary", "") or "").strip()
    if decision not in {"accepted", "rejected", "incomplete"}:
        result["reason"] = 'decision must be "accepted", "rejected", or "incomplete"'
        result["summary"] = result["reason"]
        return result

    reqs = status_doc.get("requirements")
    if not isinstance(reqs, list) or not reqs:
        result["reason"] = "requirements must be a non-empty list"
        result["summary"] = result["reason"]
        return result

    seen_ids: set[str] = set()
    normalized_items: List[Dict[str, Any]] = []
    has_fail = False
    has_unknown = False

    for idx, req in enumerate(reqs):
        if not isinstance(req, dict):
            result["reason"] = f"requirements[{idx}] must be an object"
            result["summary"] = result["reason"]
            return result

        rid = str(req.get("id", "") or "").strip()
        text = str(req.get("text", "") or "").strip()
        status = str(req.get("status", "") or "").strip().lower()
        evidence = str(req.get("evidence", "") or "").strip()

        if rid not in canonical:
            result["reason"] = f"requirements[{idx}].id is not defined in ACCEPTANCE_ITEMS.json: {rid}"
            result["summary"] = result["reason"]
            return result
        if rid in seen_ids:
            result["reason"] = f"duplicate requirement in status: {rid}"
            result["summary"] = result["reason"]
            return result
        seen_ids.add(rid)

        expected_text = canonical[rid]["text"]
        if text != expected_text:
            result["reason"] = f"requirements[{idx}].text does not match canonical acceptance text for {rid}"
            result["summary"] = result["reason"]
            return result

        if status not in {"pass", "fail", "unknown"}:
            result["reason"] = f'requirements[{idx}].status must be "pass", "fail", or "unknown"'
            result["summary"] = result["reason"]
            return result
        if status in {"pass", "fail"} and not evidence:
            result["reason"] = f"requirements[{idx}].evidence is required when status is {status}"
            result["summary"] = result["reason"]
            return result

        if status == "pass":
            result["counts"]["pass"] += 1
        elif status == "fail":
            result["counts"]["fail"] += 1
            has_fail = True
        else:
            result["counts"]["unknown"] += 1
            has_unknown = True

        normalized_items.append({"id": rid, "text": text, "status": status, "evidence": evidence})

    missing_ids = [rid for rid in canonical.keys() if rid not in seen_ids]
    if missing_ids:
        result["reason"] = "acceptance status is missing requirements: " + ", ".join(missing_ids)
        result["summary"] = result["reason"]
        return result

    artifacts = status_doc.get("artifacts")
    if artifacts is not None:
        if not isinstance(artifacts, list):
            result["reason"] = "artifacts must be a list if provided"
            result["summary"] = result["reason"]
            return result
        for idx, raw in enumerate(artifacts):
            path = resolve_artifact_path(paths, raw)
            if path is None:
                result["reason"] = f"artifacts[{idx}] is empty"
                result["summary"] = result["reason"]
                return result
            if not path.exists():
                result["reason"] = f"artifacts[{idx}] does not exist: {raw}"
                result["summary"] = result["reason"]
                return result

    result["items"] = normalized_items

    if decision == "accepted":
        if has_fail or has_unknown:
            result["reason"] = 'decision=accepted requires every requirement status to be "pass"'
            result["summary"] = result["reason"]
            return result
        result["ok"] = True
        if not result["summary"]:
            result["summary"] = "all acceptance requirements passed"
        return result

    if decision == "rejected":
        if has_unknown:
            result["reason"] = 'decision=rejected requires every requirement to be evaluated as "pass" or "fail"'
            result["summary"] = result["reason"]
            return result
        if not has_fail:
            result["reason"] = "decision=rejected requires at least one failed requirement"
            result["summary"] = result["reason"]
            return result
        result["ok"] = True
        if not result["summary"]:
            result["summary"] = "task ended with unmet acceptance requirements"
        return result

    result["reason"] = "acceptance evaluation is incomplete"
    if not result["summary"]:
        result["summary"] = result["reason"]
    return result


def acceptance_status_for_prompt(paths: AgentPaths) -> Dict[str, Any]:
    gate = evaluate_acceptance_gate(paths)
    items = gate.get("items") or []

    outstanding: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    passed: List[Dict[str, Any]] = []

    for item in items:
        status = item.get("status")
        if status == "pass":
            passed.append(item)
        elif status == "fail":
            failed.append(item)
        else:
            outstanding.append(item)

    return {
        "gate": gate,
        "passed": passed,
        "failed": failed,
        "outstanding": outstanding,
    }


def render_requirement_list(items: List[Dict[str, Any]], *, include_evidence: bool = False, empty_text: str = "(none)") -> str:
    if not items:
        return empty_text
    lines: List[str] = []
    for item in items:
        line = f"- [{item.get('id')}] {item.get('text')}"
        if include_evidence:
            evidence = str(item.get("evidence", "") or "").strip()
            if evidence:
                line += f" | evidence: {evidence}"
        lines.append(line)
    return "\n".join(lines)


def build_acceptance_repair_prompt(*, paths: AgentPaths, extra_user_prompt: str = "") -> str:
    state = acceptance_status_for_prompt(paths)
    gate = state["gate"]
    task_text = paths.task_snapshot_path.read_text(encoding="utf-8", errors="replace").strip()
    acceptance_text = paths.acceptance_snapshot_path.read_text(encoding="utf-8", errors="replace").strip()

    prompt = f"""Perform one acceptance repair / evidence completion pass.

The task cannot end yet because the acceptance gate has not passed:
- reason: {gate.get('reason') or gate.get('summary') or 'acceptance gate not satisfied'}

Current agent namespace:
- agent_id: {paths.agent_id}
- spec_dir: {paths.spec_dir}
- state_dir: {paths.state_dir}

You should now prioritize the following:
1. Read the task snapshot, acceptance snapshot, `{paths.progress_path}`, and recent run logs to confirm the real state.
2. Prioritize unfinished or failed acceptance items and obtain direct evidence.
3. If necessary, fill in small missing changes; do not redo the entire task without bounds.
4. Truthfully update `{paths.acceptance_status_path}`:
   - `decision`: accepted | rejected | incomplete
   - `summary`
   - `requirements`: each item must include at least `id`, `text`, `status`, and `evidence`
   - `artifacts` / `notes` / `next_steps` are optional
5. Update `{paths.progress_path}` and clearly state what was verified in this round and what is still missing.

Current outstanding acceptance items:
{render_requirement_list(state['outstanding'])}

Current failed acceptance items:
{render_requirement_list(state['failed'], include_evidence=True)}

Hard constraints:
- Without evidence, do not mark a requirement as pass or fail.
- Do not modify `{paths.task_source_path}`, `{paths.acceptance_source_path}`, `{paths.task_snapshot_path}`, or `{paths.acceptance_snapshot_path}` as a way to "complete the task."
- Task-related project files may be modified anywhere in the workspace, but do not modify any files under other agent namespaces.
- If the task is genuinely impossible to complete, you may write `rejected`, but you must explain the failure evidence item by item.
- Do not fake `accepted`.

## TASK SNAPSHOT
{task_text}

## ACCEPTANCE SNAPSHOT
{acceptance_text}
"""
    extra_user_prompt = (extra_user_prompt or "").strip()
    if extra_user_prompt:
        prompt += f"\n\nAdditional AI hint (only to assist acceptance repair; do not drift beyond this round's scope):\n{extra_user_prompt}\n"
    prompt += "\nExecute now. Do not respond with only a plan."
    return prompt


def synthesize_final_message(paths: AgentPaths) -> str:
    gate = evaluate_acceptance_gate(paths)
    data = gate.get("data") or {}
    decision = str(data.get("decision", "") or "").strip().lower()
    summary = str(data.get("summary", "") or "").strip()
    items = gate.get("items") or []
    artifacts = data.get("artifacts") or []

    lines: List[str] = []
    if decision == "accepted":
        lines.append("Task completed: all acceptance items have passed.")
    elif decision == "rejected":
        lines.append("Task ended with an honest failure: at least one acceptance item did not pass.")
    else:
        lines.append("The task has not yet reached an endable state.")

    lines.append(f"agent_id: {paths.agent_id}")
    lines.append(f"spec_dir: {paths.spec_dir}")
    lines.append(f"state_dir: {paths.state_dir}")
    lines.append(f"ACCEPTANCE_STATUS: {paths.acceptance_status_path}")
    if summary:
        lines.append(f"Summary: {summary}")

    if items:
        lines.append("Requirements:")
        for item in items:
            text = str(item.get("text", "") or "").strip()
            status = str(item.get("status", "") or "unknown").strip().lower()
            evidence = str(item.get("evidence", "") or "").strip()
            line = f"- [{item.get('id')}] [{status}] {text}"
            if evidence:
                line += f" | evidence: {evidence}"
            lines.append(line)

    if artifacts:
        lines.append("Artifacts:")
        for item in artifacts:
            lines.append(f"- {item}")

    return "\n".join(lines)


# -------------------------
# OpenAI helpers
# -------------------------

def extract_response_text(resp: Any) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t

    out = getattr(resp, "output", None) or []
    chunks: List[str] = []
    for item in out:
        itype = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
        if itype != "message":
            continue
        content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
        if not content:
            continue
        for c in content:
            ctype = getattr(c, "type", None) if not isinstance(c, dict) else c.get("type")
            if ctype == "output_text":
                text = getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks)


def dump_response_json(resp: Any) -> str:
    try:
        fn = getattr(resp, "model_dump", None)
        if callable(fn):
            return json.dumps(fn(), ensure_ascii=False, indent=2)
    except Exception:
        pass
    try:
        fn = getattr(resp, "dict", None)
        if callable(fn):
            return json.dumps(fn(), ensure_ascii=False, indent=2)
    except Exception:
        pass
    return json.dumps(
        {
            "id": getattr(resp, "id", None),
            "model": getattr(resp, "model", None),
            "output_text": getattr(resp, "output_text", None),
            "output": getattr(resp, "output", None),
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def call_ai(client: Any, model: str, context: str) -> Tuple[str, Any]:
    resp = client.responses.create(
        model=model,
        instructions=AI_INSTRUCTIONS,
        input=[{"role": "user", "content": context}],
        truncation="auto",
    )
    return extract_response_text(resp).strip(), resp


# -------------------------
# AI output parsing
# -------------------------

def parse_ai_output(text: str) -> Tuple[str, str]:
    t = (text or "").strip()
    m = ACTION_RE.search(t)
    action = (m.group(1).upper() if m else "RUN_CODEX")

    def block(name: str) -> Optional[str]:
        pat = re.compile(
            rf"^BEGIN_{name}\s*$\n(.*?)\n^END_{name}\s*$",
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        mm = pat.search(t)
        return mm.group(1).strip() if mm else None

    if action == "RUN_CODEX":
        b = block("CODEX")
        if b:
            return action, b
        t2 = ACTION_RE.sub("", t, count=1).strip()
        return action, t2 if t2 else t

    if action == "ASK_HUMAN":
        b = block("QUESTION")
        if b:
            return action, b
        t2 = ACTION_RE.sub("", t, count=1).strip()
        return action, t2 if t2 else "(AI did not provide a question)"

    if action == "FINISH":
        b = block("FINAL")
        if b:
            return action, b
        t2 = ACTION_RE.sub("", t, count=1).strip()
        return action, t2 if t2 else t

    return "RUN_CODEX", t


# -------------------------
# Codex runner
# -------------------------

def tail_bytes_with_gap(path: Path, max_bytes: int) -> Dict[str, Any]:
    if not path.exists():
        return {"tail": "", "total_bytes": 0, "tail_bytes": 0, "missing_bytes": 0}

    total = path.stat().st_size
    if total <= 0:
        return {"tail": "", "total_bytes": 0, "tail_bytes": 0, "missing_bytes": 0}

    want = max(0, int(max_bytes))
    if want <= 0:
        return {"tail": "", "total_bytes": total, "tail_bytes": 0, "missing_bytes": total}

    with path.open("rb") as f:
        if total <= want:
            data = f.read()
        else:
            f.seek(total - want)
            data = f.read()

    tail_b = len(data)
    missing = max(0, total - tail_b)
    tail_txt = data.decode("utf-8", errors="replace")
    return {"tail": tail_txt, "total_bytes": total, "tail_bytes": tail_b, "missing_bytes": missing}


def build_codex_preamble(paths: AgentPaths, mindset_text: str) -> str:
    state = acceptance_status_for_prompt(paths)
    passed = state["passed"]
    failed = state["failed"]
    outstanding = state["outstanding"]

    task_text = read_text_limited(paths.task_snapshot_path, 2500, tail=False)
    acceptance_text = read_text_limited(paths.acceptance_snapshot_path, 2500, tail=False)

    lines: List[str] = []
    lines.append(f"AGENT NAMESPACE:")
    lines.append(f"- agent_id: {paths.agent_id}")
    lines.append(f"- spec_dir: {paths.spec_dir}")
    lines.append(f"- state_dir: {paths.state_dir}")
    lines.append("")
    lines.append("Note: project files may be modified anywhere in the workspace as required by the task; the agent's own bookkeeping may only be written inside this namespace.")
    lines.append("")
    lines.append("MINDSET (must be followed; from spec/mindset.md):")
    lines.append((mindset_text or "(empty)").strip())
    lines.append("")
    lines.append("TASK SNAPSHOT (do not reinterpret or rewrite its meaning):")
    lines.append(task_text.strip() or "(empty)")
    lines.append("")
    lines.append("ACCEPTANCE SNAPSHOT (this is the contract for ending, not a suggestion):")
    lines.append(acceptance_text.strip() or "(empty)")
    lines.append("")
    lines.append("ACCEPTANCE STATUS SUMMARY:")
    lines.append(f"- pass: {len(passed)}")
    lines.append(f"- fail: {len(failed)}")
    lines.append(f"- unknown: {len(outstanding)}")
    lines.append("")
    lines.append("OUTSTANDING ACCEPTANCE ITEMS:")
    lines.append(render_requirement_list(outstanding))
    lines.append("")
    lines.append("FAILED ACCEPTANCE ITEMS:")
    lines.append(render_requirement_list(failed, include_evidence=True))
    lines.append("")
    lines.append("Execution rules (mandatory):")
    lines.append("1) At the start of each round, inspect the outstanding acceptance items first and prioritize them.")
    lines.append("2) Prioritize real evidence; when commands, tests, interfaces, or files are involved, inspect the actual outputs first.")
    lines.append(f"3) Update {paths.progress_path} every round, clearly stating the changes, verification, and remaining work.")
    lines.append(f"4) Whenever the status of an acceptance item changes, promptly and truthfully update {paths.acceptance_status_path}.")
    lines.append("5) Promptly clean up useless temporary files, debug output, or caches that you created; if cleanup itself is a key acceptance condition, it must be reflected in acceptance evidence.")
    lines.append(f"6) Do not fake completion by modifying {paths.task_source_path}, {paths.acceptance_source_path}, {paths.task_snapshot_path}, or {paths.acceptance_snapshot_path}.")
    lines.append("7) Do not modify any files under other agent namespaces unless the task explicitly requires it.")
    lines.append("8) Without evidence, do not mark an acceptance item as pass.")
    return "\n".join(lines)


def run_codex_exec(*, workspace: Path, codex_model: str, codex_prompt: str, run_dir: Path, timeout_s: int, log_tail_bytes: int) -> Dict[str, Any]:
    if shutil.which("codex") is None:
        raise FileNotFoundError("codex CLI not found in PATH.")

    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_p = run_dir / "codex.stdout.txt"
    stderr_p = run_dir / "codex.stderr.txt"

    cmd: List[str] = [
        "codex",
        "exec",
        "-C",
        str(workspace),
        "-s",
        "workspace-write",
        "--skip-git-repo-check",
        "--color",
        "never",
        "-c",
        f'model="{codex_model}"',
        "-c",
        'approval_policy="never"',
        "-c",
        'sandbox_workspace_write.network_access=true',
        "--",
        codex_prompt,
    ]

    with stdout_p.open("w", encoding="utf-8") as fo, stderr_p.open("w", encoding="utf-8") as fe:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace),
                text=True,
                stdout=fo,
                stderr=fe,
                timeout=None if timeout_s <= 0 else timeout_s,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124
            fe.write("\n[TIMEOUT]\n")

    out_stats = tail_bytes_with_gap(stdout_p, log_tail_bytes)
    err_stats = tail_bytes_with_gap(stderr_p, log_tail_bytes)

    return {
        "returncode": rc,
        "cmd": cmd,
        "stdout_path": str(stdout_p),
        "stderr_path": str(stderr_p),
        "stdout_tail": out_stats["tail"],
        "stderr_tail": err_stats["tail"],
        "stdout_missing_bytes": out_stats["missing_bytes"],
        "stderr_missing_bytes": err_stats["missing_bytes"],
    }


# -------------------------
# Context building
# -------------------------

def build_ai_context(*, paths: AgentPaths, round_idx: int, max_rounds: int, codex_model: str, codex_timeout_s: int, mindset_text: str, history: List[Dict[str, str]]) -> str:
    rounds_left = max(0, max_rounds - round_idx)
    state = acceptance_status_for_prompt(paths)
    gate = state["gate"]

    parts: List[str] = []
    parts.append(f"## AGENT NAMESPACE\n- agent_id: {paths.agent_id}\n- agent_home: {paths.agent_home}\n- spec_dir: {paths.spec_dir}\n- state_dir: {paths.state_dir}")
    parts.append("\n## TASK SNAPSHOT\n" + read_text_limited(paths.task_snapshot_path, 6000, tail=False).strip())
    parts.append("\n## ACCEPTANCE SNAPSHOT\n" + read_text_limited(paths.acceptance_snapshot_path, 6000, tail=False).strip())
    parts.append("\n## MINDSET (authoritative; cross-task principles only)\n" + (mindset_text or "(empty)"))

    parts.append("\n## ENV / CONSTRAINTS")
    parts.append(f"- workspace: {paths.workspace}")
    parts.append(f"- workspace may contain multiple agent namespaces under {paths.agents_root}")
    parts.append("- only this agent namespace should be mutated for bookkeeping")
    parts.append("- loop: AI <-> Codex alternating")
    parts.append(f"- round_idx: {round_idx} / max_rounds: {max_rounds} (rounds_left={rounds_left})")
    parts.append(f"- codex_model: {codex_model}")
    parts.append(f"- codex_timeout_s: {codex_timeout_s}")
    parts.append('- approvals disabled via: -c approval_policy="never"')
    parts.append("- codex prompt is passed as positional arg after `--`")

    parts.append("\n## SOURCE OF TRUTH")
    parts.append(f"- task_source: {paths.task_source_path}")
    parts.append(f"- acceptance_source: {paths.acceptance_source_path}")
    parts.append(f"- mindset_source: {paths.mindset_source_path}")
    parts.append("- source files may be human-maintained; do not treat editing them as task completion")

    parts.append("\n## ACCEPTANCE GATE")
    parts.append(f"- acceptance_status_path: {paths.acceptance_status_path}")
    parts.append(f"- acceptance_gate_ok: {gate.get('ok')}")
    parts.append(f"- acceptance_gate_reason: {gate.get('reason') or '(none)'}")
    if gate.get("decision"):
        parts.append(f"- acceptance_decision: {gate.get('decision')}")
    if gate.get("summary"):
        parts.append(f"- acceptance_summary: {gate.get('summary')}")

    counts = gate.get("counts") or {}
    parts.append("\n## ACCEPTANCE STATUS SUMMARY")
    parts.append(f"- pass: {counts.get('pass', 0)}")
    parts.append(f"- fail: {counts.get('fail', 0)}")
    parts.append(f"- unknown: {counts.get('unknown', 0)}")

    parts.append("\n## OUTSTANDING ACCEPTANCE ITEMS")
    parts.append(render_requirement_list(state["outstanding"]))

    parts.append("\n## FAILED ACCEPTANCE ITEMS")
    parts.append(render_requirement_list(state["failed"], include_evidence=True))

    status_text = read_text_limited(paths.acceptance_status_path, 5000, tail=False)
    if status_text:
        parts.append(f"\n## {paths.acceptance_status_path}\n" + status_text)

    parts.append("\n## MEMORY (tails)")
    parts.append(f"### {paths.progress_path} (tail)\n" + read_text_limited(paths.progress_path, DEFAULT_CONTEXT_TAIL_CHARS, tail=True))
    parts.append(f"\n### {paths.human_path} (tail)\n" + read_text_limited(paths.human_path, 1800, tail=True))

    parts.append("\n## RECENT HISTORY (most recent last)")
    if history:
        for h in history[-10:]:
            role = h.get("role", "unknown")
            content = h.get("content", "")
            parts.append(f"\n[{role}]\n{content}\n")
    else:
        parts.append("(none yet)\n")

    parts.append("\nNow output the next step. Prefer the smallest next step that reduces outstanding or failed acceptance items.")
    return "\n".join(parts)


# -------------------------
# main
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Path to the shared workspace directory.")
    ap.add_argument(
        "--agent-id",
        default=DEFAULT_AGENT_ID,
        help=f'Agent namespace under <workspace>/{AGENTS_DIRNAME}/<agent-id> (default "{DEFAULT_AGENT_ID}").',
    )
    ap.add_argument(
        "--task",
        default=None,
        help="Task source: inline text or @path. If omitted, use this agent namespace's spec/task.md.",
    )
    ap.add_argument(
        "--acceptance",
        default=None,
        help="Acceptance source: inline text or @path. If omitted, use this agent namespace's spec/acceptance.md.",
    )
    ap.add_argument(
        "--mindset",
        default=None,
        help="Optional mindset source: inline text or @path. If omitted, use the built-in default engineering mindset.",
    )
    ap.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help=f"Safety cap for AI<->Codex rounds (default {DEFAULT_MAX_ROUNDS}).",
    )
    ap.add_argument(
        "--ai-model",
        default=DEFAULT_AI_MODEL,
        help=f"Planner model for OpenAI Responses (default {DEFAULT_AI_MODEL}).",
    )
    ap.add_argument(
        "--codex-model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Model passed to codex exec (default {DEFAULT_CODEX_MODEL}).",
    )
    ap.add_argument(
        "--codex-timeout-s",
        type=int,
        default=DEFAULT_CODEX_TIMEOUT_S,
        help=f"Timeout for one codex exec round in seconds (default {DEFAULT_CODEX_TIMEOUT_S}).",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help="Delete only this agent namespace's state/ directory before starting; preserve spec/.",
    )

    args = ap.parse_args()

    if int(args.max_rounds) <= 0:
        print("max-rounds must be positive", file=sys.stderr)
        return 2
    if int(args.codex_timeout_s) <= 0:
        print("codex-timeout-s must be positive", file=sys.stderr)
        return 2

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists() or not workspace.is_dir():
        print(f"Workspace not found or not a directory: {workspace}", file=sys.stderr)
        return 2

    try:
        paths = build_paths(workspace, args.agent_id)
    except Exception as e:
        print(f"Invalid agent-id: {e}", file=sys.stderr)
        return 2

    if args.reset:
        reset_agent_state(paths)

    ensure_layout(paths)

    try:
        source_meta = initialize_namespace_sources(
            paths=paths,
            workspace=workspace,
            task_raw=args.task,
            acceptance_raw=args.acceptance,
            mindset_raw=args.mindset,
        )
        write_run_meta(paths, source_meta)
        sync_spec_files(paths)
    except Exception as e:
        print(f"Input/setup error: {e}", file=sys.stderr)
        return 2

    append_progress(paths, "[System] Initialized namespace sources, snapshots, and acceptance status")

    try:
        from openai import OpenAI
    except Exception as e:
        print(f"Dependency error: openai package is required ({e})", file=sys.stderr)
        return 2

    client = OpenAI()
    history: List[Dict[str, str]] = []

    def load_mindset() -> str:
        return read_text_limited(paths.mindset_source_path, DEFAULT_MINDSET_MAX_CHARS, tail=False).strip()

    def refresh_truth() -> None:
        sync_spec_files(paths)

    def call_ai_once(run_dir: Path, *, round_idx: int) -> Tuple[str, str, str]:
        refresh_truth()
        mindset_text = load_mindset()
        ctx = build_ai_context(
            paths=paths,
            round_idx=round_idx,
            max_rounds=int(args.max_rounds),
            codex_model=args.codex_model,
            codex_timeout_s=int(args.codex_timeout_s),
            mindset_text=mindset_text,
            history=history,
        )

        ai_text, ai_resp = call_ai(client, args.ai_model, ctx)

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ai.context.txt").write_text(ctx, encoding="utf-8")
        (run_dir / "ai.output.txt").write_text(ai_text, encoding="utf-8")
        (run_dir / "ai.response.json").write_text(dump_response_json(ai_resp), encoding="utf-8")

        action, payload = parse_ai_output(ai_text)
        return ai_text, action, payload

    def record_ai_to_history(ai_text: str, action: str, *, tag: str = "AI") -> None:
        append_progress(paths, f"[{tag}] action={action} {one_line(ai_text)}")
        history.append({"role": tag, "content": ai_text})

    def run_codex_round(run_dir: Path, codex_prompt_user: str) -> Dict[str, Any]:
        refresh_truth()
        codex_prompt_user = (codex_prompt_user or "").strip()
        if not codex_prompt_user:
            codex_prompt_user = (
                "SENSE FIRST.\n"
                "1) Inspect the workspace.\n"
                "2) Identify the smallest next verifiable step.\n"
                f"3) Update {paths.progress_path}.\n"
                f"4) Update {paths.acceptance_status_path} if any acceptance evidence changed.\n"
            )

        mindset_text = load_mindset()
        codex_prompt_effective = build_codex_preamble(paths, mindset_text) + "\n\n" + codex_prompt_user

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "codex.prompt.user.txt").write_text(codex_prompt_user, encoding="utf-8")
        (run_dir / "codex.prompt.txt").write_text(codex_prompt_effective, encoding="utf-8")
        append_progress(paths, "[Codex] Starting codex exec")

        res = run_codex_exec(
            workspace=workspace,
            codex_model=args.codex_model,
            codex_prompt=codex_prompt_effective,
            run_dir=run_dir,
            timeout_s=int(args.codex_timeout_s),
            log_tail_bytes=DEFAULT_LOG_TAIL_BYTES,
        )

        append_progress(paths, f"[Codex] Done (returncode={res['returncode']})")

        codex_result = json.dumps(
            {
                "returncode": res["returncode"],
                "cmd": res["cmd"],
                "stdout_path": res["stdout_path"],
                "stderr_path": res["stderr_path"],
                "stdout_tail": res["stdout_tail"],
                "stderr_tail": res["stderr_tail"],
                "stdout_missing_bytes": res["stdout_missing_bytes"],
                "stderr_missing_bytes": res["stderr_missing_bytes"],
                "hint": "If the tail is insufficient, ask Codex to read the full log files from the saved paths.",
            },
            ensure_ascii=False,
        )
        (run_dir / "codex.result.json").write_text(codex_result, encoding="utf-8")
        history.append({"role": "CODEX_RESULT", "content": codex_result})
        return res

    def maybe_accept_finish(final_msg: str) -> Optional[int]:
        refresh_truth()
        gate = evaluate_acceptance_gate(paths)
        if gate["ok"]:
            append_progress(paths, "[AI] FINISH")
            print(final_msg if final_msg else synthesize_final_message(paths))
            return 0
        return None

    for round_idx in range(int(args.max_rounds)):
        run_dir = paths.runs_dir / f"round_{round_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            ai_text, action, payload = call_ai_once(run_dir, round_idx=round_idx)
        except Exception as e:
            append_progress(paths, f"[System] AI call failed: {e}")
            print(f"AI error: {e}", file=sys.stderr)
            return 3

        record_ai_to_history(ai_text, action)

        if action == "ASK_HUMAN":
            q = payload.strip()
            print("\n[AI asks human]")
            print(q)
            try:
                ans = input("> ").strip()
            except EOFError:
                ans = ""
            append_human(paths, q, ans)
            append_progress(paths, f"[HUMAN] {one_line(ans)}")
            history.append({"role": "HUMAN", "content": ans})
            continue

        if action == "FINISH":
            accepted = maybe_accept_finish(payload.strip())
            if accepted is not None:
                return accepted

            gate = evaluate_acceptance_gate(paths)
            append_progress(paths, f"[System] Blocked FINISH: {gate['reason']}")
            history.append(
                {
                    "role": "SYSTEM",
                    "content": (
                        "FINISH blocked because the acceptance gate did not pass. "
                        f"Reason: {gate['reason']}. "
                        f"You must update {paths.acceptance_status_path} truthfully before finishing."
                    ),
                }
            )
            action = "RUN_CODEX"
            payload = build_acceptance_repair_prompt(paths=paths, extra_user_prompt="")

        try:
            run_codex_round(run_dir, payload.strip())
        except Exception as e:
            append_progress(paths, f"[Codex] ERROR: {e}")
            history.append({"role": "CODEX_RESULT", "content": f"ERROR: {e}"})
            continue

    refresh_truth()
    gate = evaluate_acceptance_gate(paths)
    if gate["ok"]:
        append_progress(paths, f"[System] Reached max_rounds={args.max_rounds} with a passing acceptance gate")
        print(synthesize_final_message(paths))
        return 0

    append_progress(
        paths,
        f"[System] Reached max_rounds={args.max_rounds}; last gate reason: {gate['reason'] or gate['summary']}",
    )
    print(f"Stopped: reached max_rounds={args.max_rounds}.")
    if gate["reason"]:
        print(f"Acceptance gate reason: {gate['reason']}")
    print(f"Check {paths.acceptance_status_path} and {paths.runs_dir} for evidence.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())