"""Tests for the neoo run wrapper (run_episode orchestration).

Path-loads both neoo modules (their orchestration core is dependency-free) and drives run_episode
with a FAKE agent + an injected builder, so the end-to-end glue is verified without the agent's
runtime deps (litellm) or a live model endpoint.
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


tr = _load("neoo_trajectory", "src/minisweagent/neoo/trajectory.py")
re = _load("neoo_run", "src/minisweagent/neoo/run_episode.py")


class _FakeAgent:
    """Stands in for a mini-swe-agent DefaultAgent: .run(task) -> extra; .save(None) -> trajectory."""

    def __init__(self, trajectory, result):
        self._t, self._r, self.ran = trajectory, result, None

    def run(self, task):
        self.ran = task
        return self._r

    def save(self, path):
        return self._t


_TRAJ = {
    "info": {"exit_status": "Submitted", "submission": "diff",
             "model_stats": {"instance_cost": 0.0, "api_calls": 2}},
    "messages": [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "ok", "extra": {"actions": [{"action": "ls"}]}},
        {"role": "exit", "content": "done", "extra": {"exit_status": "Submitted", "submission": "diff"}},
    ],
}


def test_run_episode_scored_and_exported(tmp_path):
    exp = tr.EpisodeExporter(tmp_path / "ep.jsonl")
    agent = _FakeAgent(_TRAJ, {"exit_status": "Submitted", "submission": "diff"})
    rec = re.run_episode(agent, "fix the bug", task_id="repo__1",
                         reward_fn=lambda traj, result: 1.0, exporter=exp, model_id="glm-5.2")
    assert agent.ran == "fix the bug"                         # the agent actually ran the task
    assert rec["outcome"]["reward"] == 1.0 and rec["outcome"]["scored"] is True
    assert rec["task_id"] == "repo__1" and rec["model_id"] == "glm-5.2"
    line = json.loads((tmp_path / "ep.jsonl").read_text(encoding="utf-8").strip())
    assert line["episode_id"] == rec["episode_id"]           # the labeled episode was written


def test_run_episode_unscored_by_default(tmp_path):
    exp = tr.EpisodeExporter(tmp_path / "ep.jsonl")
    rec = re.run_episode(_FakeAgent(_TRAJ, {}), "t", task_id="t1", exporter=exp)  # no reward_fn
    assert rec["outcome"]["reward"] is None and rec["outcome"]["scored"] is False


def test_run_episode_no_exporter_uses_injected_builder():
    rec = re.run_episode(_FakeAgent(_TRAJ, {}), "t", task_id="t1",
                         reward_fn=lambda traj, result: 0.0, _build_episode=tr.build_episode)
    assert rec["outcome"]["reward"] == 0.0 and rec["outcome"]["scored"] is True
    assert rec["n_turns"] == 2                               # user + assistant (exit excluded)
