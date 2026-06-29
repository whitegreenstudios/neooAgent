"""
neoo.run_episode — one command: solve a task with neooAgent and emit a LABELED episode
(verifiable-reward RL training data). Ties together neoo_model() (served GLM-5.2) + the agent +
the reward source + EpisodeExporter into a single end-to-end entry point for the flywheel.

    python -m minisweagent.neoo.run_episode --task "Fix the bug in foo()" --task-id repo__issue-1

All `minisweagent` imports are LAZY (inside functions) so the orchestration core (`run_episode`) is
importable + testable without the agent's runtime deps (litellm etc.).

Reward source: pluggable `reward_fn`. The verifiable reward must come from the repo-level test
checker (NeuraHash #65) — never the agent grading itself. Until #65 is wired, episodes export
UNSCORED (outcome.scored=False) so the data is still collected and labeled later.
"""


def run_episode(agent, task, *, task_id, reward_fn=None, exporter=None, model_id="",
                _build_episode=None):
    """Run `agent` on `task`, score the run, export a labeled episode. Returns the episode record.

    agent     : anything with .run(task) and .save(None) (a mini-swe-agent DefaultAgent or a stub)
    reward_fn : (trajectory, result) -> float | None — the VERIFIABLE reward source (the #65
                repo-test checker). None -> the episode is exported UNSCORED until #65 is wired.
    exporter  : an EpisodeExporter (appends JSONL). If None, the record is built and returned only.
    """
    result = agent.run(task)
    trajectory = agent.save(None)
    reward = reward_fn(trajectory, result) if reward_fn is not None else None
    if exporter is not None:
        return exporter.export(trajectory, reward=reward, task_id=task_id, model_id=model_id)
    build = _build_episode or _load_build_episode()
    return build(trajectory, reward=reward, task_id=task_id, model_id=model_id)


def _load_build_episode():
    from minisweagent.neoo.trajectory import build_episode

    return build_episode


def build_agent(*, config_name="mini.yaml", model=None, env=None, agent_overrides=None):
    """Construct a neooAgent wired to our served model, reusing mini-swe-agent's factories + a
    builtin config (for the prompt templates). Needs the agent runtime deps installed and (for live
    runs) NEOO_API_BASE pointing at our serve endpoint. (A dedicated `swebench.yaml` config for the
    SWE-bench harness is a later increment; `mini.yaml` is the default local config.)"""
    from minisweagent.agents import get_agent
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.environments import get_environment
    from minisweagent.neoo.model import neoo_model
    from minisweagent.utils.serialize import recursive_merge

    cfg = get_config_from_spec(str(builtin_config_dir / config_name))
    model = model or neoo_model()
    env = env or get_environment(cfg.get("environment", {}), default_type="local")
    agent_cfg = recursive_merge(cfg.get("agent", {}), agent_overrides or {})
    return get_agent(model, env, agent_cfg, default_type="default")


def solve(task, *, task_id, reward_fn=None, out_path="neoo_episodes.jsonl", model_id=None,
          config_name="mini.yaml", **build_kwargs):
    """High-level: build a served-model agent, solve `task`, export a labeled episode to `out_path`.
    Returns the episode record."""
    import os

    from minisweagent.neoo.trajectory import EpisodeExporter

    agent = build_agent(config_name=config_name, **build_kwargs)
    mid = model_id if model_id is not None else os.getenv("NEOO_MODEL_NAME", "")
    return run_episode(agent, task, task_id=task_id, reward_fn=reward_fn,
                       exporter=EpisodeExporter(out_path), model_id=mid)


def _cli():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="neooAgent: solve a task and emit a labeled RL episode")
    ap.add_argument("--task", required=True, help="the issue / problem statement")
    ap.add_argument("--task-id", required=True, help="provenance id (e.g. the SWE-bench instance id)")
    ap.add_argument("--config", default="mini.yaml", help="builtin config name (prompt templates)")
    ap.add_argument("--out", default="neoo_episodes.jsonl", help="JSONL output path")
    a = ap.parse_args()
    rec = solve(a.task, task_id=a.task_id, out_path=a.out, config_name=a.config)
    o = rec["outcome"]
    print(json.dumps({"episode_id": rec["episode_id"], "scored": o["scored"], "reward": o["reward"],
                      "exit_status": o["exit_status"], "out": a.out}))


if __name__ == "__main__":
    _cli()
