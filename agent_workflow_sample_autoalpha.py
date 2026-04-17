#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from agent_workflow import Agent, run_workflow

# -----------------------------------------------------------------------------
# Minimal alpha-mining / factor-testing sample
#
# Design goals:
# 1. Keep long-lived idea storage and per-run candidate selection separate.
# 2. Keep terminal test results and in-progress attempt logs separate.
# 3. Let the agent choose the next idea from the current run manifest.
# 4. Keep the workflow graph simple: A generates ideas, B tests one idea per loop.
#
# This sample assumes you are using the newer agent_workflow.py where each
# invocation gets an independent agent_id. The old fixed-id workflow runner will
# reintroduce cross-run state pollution.
# -----------------------------------------------------------------------------

WORKSPACE = Path("fill in your infra workspace path here")
IDEA_LIST = WORKSPACE / "idea_list"

DRY_RUN = False
WORKFLOW = f"loop(A->B,10)"


def bootstrap_workspace() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not IDEA_LIST.exists():
        IDEA_LIST.write_text("# IDEA LIST\n\n", encoding="utf-8")


A_TASK = f"""
Your only responsibility is to generate new ideas. Do not perform any testing.

The key file roles for this run are:
- `{IDEA_LIST.name}`: long-term idea repository


1. Study the existing feature files and explore available data. This project is about time-series return prediction.
2. Read the local idea_list, refer to existing ideas, and propose a new feature idea. The new idea must be different from existing ideas and features (i.e., low idea correlation). Write the new idea into idea_list.
3. The new idea's ID must increment sequentially after the current maximum ID.
4. Each new idea must include the following non-empty fields:
   - name
   - raw definition
   - parameter suggestions
   - implementation details

Constraints:
- Do not test ideas.
- Do not delete or modify existing ideas or their business meaning.
- Do not describe something as "proven effective" if it is only hypothetical.
""".strip()

A_ACCEPTANCE = f"""
- 1. A new idea has been written to idea_list, and only the new idea was added
- 2. No other work artifacts were left behind
""".strip()

B_TASK = f"""

Primary objective:
Test the idea with the largest ID in the local idea_list. Use the default time range for full evaluation.
You may tune parameters and certain conditions (without changing the idea itself) to improve feature performance.

Acceptance criteria:
1. Real full backtest.py results, satisfying:
   - ic >= 0.06
2. Real full corr.py results, satisfying:
   - maximum |corr| with existing features <= 0.4
3. Additional requirements can be added as needed.

Cleanup hard constraints:
1. Cleaning temporary files is a hard requirement, not optional.
2. After each failed attempt, temporary files must be deleted immediately; do not defer all cleanup to the final round.
3. Before finishing, you must write `.agent/FINISH_CHECK.json`, including at least:
   - task_status
   - full_test_verified
   - ic_verified
   - corr_verified
   - cleanup_verified
   - improvement_space_exhausted
   - kept_feature_paths
   - before_listing_artifact
   - after_listing_artifact
   - notes

Completion requirements:
1. Before finishing, remove all temporary files except for features that passed the test.
2. If the feature does not meet the criteria, you must continue improving it (e.g., parameter tuning) until no further improvement is possible.
3. If the task is not completed, do not pretend it succeeded.
4. If no candidate passes within the round limit, do not fake success; terminate truthfully as failed_no_candidate, while still completing cleanup and leaving evidence.
""".strip()

B_ACCEPTANCE = f"""
The task is accepted if all of the following 4 conditions are met:
1. The tested object is the idea with the largest ID in idea_list
2. Full evaluation is performed using the default time range, and results satisfy:
   - ic >= 0.06
   - maximum |corr| with existing features <= 0.4
   - other requirements
3. The runs directory contains both pre-cleanup and post-cleanup listings
4. Only features that passed the test are retained; all other temporary files are cleaned up; `.agent/FINISH_CHECK.json` accurately reflects the task status

""".strip()

A = Agent(
    name="A",
    script="./agent.py",
    workspace=WORKSPACE,
    task=A_TASK,
    acceptance=A_ACCEPTANCE,
    max_rounds=6,
    # reset=False,
    dry_run=DRY_RUN,
)

B = Agent(
    name="B",
    script="./agent.py",
    workspace=WORKSPACE,
    task=B_TASK,
    acceptance=B_ACCEPTANCE,
    max_rounds=10,
    # reset=False,
    dry_run=DRY_RUN,
)

AGENTS = [A, B]


if __name__ == "__main__":
    bootstrap_workspace()
    run_id = run_workflow(WORKFLOW, AGENTS)
    print(f"done: {WORKSPACE} workflow_run_id={run_id}")
