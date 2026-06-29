"""
neoo.reward — bridge an agent's SUBMISSION (its patch) to a verifiable reward checker.

`run_episode(reward_fn=...)` wants `reward_fn(trajectory, result) -> float | None`. The verifiable
reward comes from the repo-level test-execution checker on the NeuraHash side
(`neurahash.repo_check.RepoTestExec`, issue #65): apply the patch, run the repo's tests, 1.0/0.0.

We keep neooAgent free of the neurahash dependency by INJECTING the checker (any callable
`patch -> float`), so this module is the thin, dependency-free seam between the two repos:

    from neurahash.repo_check import from_spec, SubprocessExecutor   # NeuraHash side (the #65 checker)
    from minisweagent.neoo import run_episode, EpisodeExporter
    from minisweagent.neoo.reward import submission_reward_fn

    checker = from_spec(task_spec, SubprocessExecutor(repo_dir))     # rebuild the pinned env
    run_episode(agent, issue, task_id=instance_id,
                reward_fn=submission_reward_fn(checker),
                exporter=EpisodeExporter("episodes.jsonl"))

The reward MUST be the program checker's verdict — never the agent grading itself. Until a real
checker is injected, episodes stay UNSCORED (pass `reward_fn=None`).
"""


def extract_submission(trajectory, result):
    """Pull the agent's submitted patch/diff. Prefer the run `result`'s `submission`, else the
    serialized `trajectory`'s `info.submission`. Returns "" if nothing was submitted."""
    if isinstance(result, dict) and result.get("submission"):
        return result["submission"]
    info = trajectory.get("info", {}) if isinstance(trajectory, dict) else {}
    return (info or {}).get("submission", "") or ""


def submission_reward_fn(checker, *, default=None):
    """Build a `run_episode` reward_fn that scores the agent's submission with `checker`
    (a `neurahash.repo_check.RepoTestExec`, or any callable `patch -> float` in {0.0, 1.0}).
    If the agent submitted nothing, returns `default` (None -> the episode stays unscored)."""

    def reward_fn(trajectory, result):
        patch = extract_submission(trajectory, result)
        if not patch:
            return default
        return float(checker(patch))

    return reward_fn
