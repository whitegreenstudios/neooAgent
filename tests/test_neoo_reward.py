"""Tests for the neoo reward bridge (agent submission -> injected verifiable checker).

Path-loads the dependency-free neoo modules so the seam is verified without litellm / a real
checker / a live endpoint. A fake checker stands in for neurahash.repo_check.RepoTestExec.
"""
import importlib.util
import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rw = _load("neoo_reward", "src/minisweagent/neoo/reward.py")


def _checker(patch):                       # stands in for RepoTestExec: 1.0 iff the patch "fixes" it
    return 1.0 if "fixed" in patch else 0.0


def test_extract_submission_prefers_result():
    assert rw.extract_submission({"info": {"submission": "from-traj"}},
                                 {"submission": "from-result"}) == "from-result"


def test_extract_submission_falls_back_to_trajectory():
    traj = {"info": {"submission": "diff from-traj"}}
    assert rw.extract_submission(traj, {"exit_status": "X"}) == "diff from-traj"


def test_extract_submission_empty_when_none():
    assert rw.extract_submission({}, {}) == ""
    assert rw.extract_submission(None, None) == ""


def test_reward_fn_scores_submission_with_checker():
    rf = rw.submission_reward_fn(_checker)
    assert rf({}, {"submission": "diff that fixed it"}) == 1.0
    assert rf({}, {"submission": "diff broken"}) == 0.0


def test_reward_fn_default_when_no_submission():
    assert rw.submission_reward_fn(_checker, default=None)({}, {}) is None
    assert rw.submission_reward_fn(_checker, default=0.0)({}, {}) == 0.0


def test_end_to_end_scored_episode(tmp_path):
    """The full seam: agent submission -> submission_reward_fn(checker) -> a SCORED episode."""
    tr = _load("neoo_trajectory", "src/minisweagent/neoo/trajectory.py")
    run = _load("neoo_run", "src/minisweagent/neoo/run_episode.py")

    class _FakeAgent:
        def run(self, task):
            return {"exit_status": "Submitted", "submission": "diff that fixed it"}

        def save(self, path):
            return {"info": {"exit_status": "Submitted", "submission": "diff that fixed it",
                             "model_stats": {"api_calls": 1}},
                    "messages": [{"role": "user", "content": "fix"},
                                 {"role": "assistant", "content": "done"}]}

    exporter = tr.EpisodeExporter(tmp_path / "ep.jsonl")
    rec = run.run_episode(_FakeAgent(), "fix the bug", task_id="repo__1",
                          reward_fn=rw.submission_reward_fn(_checker), exporter=exporter,
                          model_id="glm-5.2")
    assert rec["outcome"]["reward"] == 1.0 and rec["outcome"]["scored"] is True
    assert json.loads((tmp_path / "ep.jsonl").read_text(encoding="utf-8").strip())["task_id"] == "repo__1"
