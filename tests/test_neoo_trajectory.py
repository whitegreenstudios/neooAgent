"""Tests for the neoo trajectory exporter.

Loads neoo/trajectory.py BY PATH (it is dependency-free) so it runs without the agent's runtime
deps. Verifies the neoo-episode-1 schema, terminal reward attachment, content-addressing, and the
JSONL exporter.
"""
import importlib.util
import json
import pathlib

_TR = pathlib.Path(__file__).resolve().parents[1] / "src" / "minisweagent" / "neoo" / "trajectory.py"
_spec = importlib.util.spec_from_file_location("neoo_trajectory", _TR)
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)


def _synthetic_traj():
    """Mimic DefaultAgent.serialize() output: info + a system/user/assistant/tool/assistant/exit run."""
    return {
        "trajectory_format": "mini-swe-agent-1.1",
        "info": {
            "exit_status": "Submitted",
            "submission": "diff --git a/x.py b/x.py",
            "model_stats": {"instance_cost": 0.12, "api_calls": 3},
        },
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix issue X"},
            {"role": "assistant", "content": "Let me look.", "extra": {"actions": [{"action": "ls"}]}},
            {"role": "tool", "content": "<returncode>0</returncode>\n<output>x.py</output>"},
            {"role": "assistant", "content": "Patching.", "extra": {"actions": [{"action": "edit x.py"}]}},
            {"role": "exit", "content": "done",
             "extra": {"exit_status": "Submitted", "submission": "diff --git a/x.py b/x.py"}},
        ],
    }


def test_build_episode_schema_reward_and_turns():
    rec = tr.build_episode(_synthetic_traj(), reward=1.0, task_id="repo__issue-1", model_id="glm-5.2")
    assert rec["schema"] == tr.SCHEMA
    assert rec["task_id"] == "repo__issue-1" and rec["model_id"] == "glm-5.2"
    assert rec["outcome"]["reward"] == 1.0
    assert rec["outcome"]["exit_status"] == "Submitted"
    assert rec["outcome"]["submission"].startswith("diff --git")
    assert rec["n_turns"] == 5                              # exit message excluded from turns
    assert any(t.get("actions") for t in rec["turns"])      # assistant actions preserved
    assert rec["meta"]["api_calls"] == 3


def test_episode_id_is_content_addressed():
    a = tr.build_episode(_synthetic_traj(), reward=1.0, task_id="t1", model_id="m")
    b = tr.build_episode(_synthetic_traj(), reward=1.0, task_id="t1", model_id="m")
    c = tr.build_episode(_synthetic_traj(), reward=1.0, task_id="t2", model_id="m")
    assert a["episode_id"] == b["episode_id"]               # deterministic
    assert a["episode_id"] != c["episode_id"]               # different task -> different id
    assert len(a["episode_id"]) == 64


def test_jsonl_roundtrip_and_exporter(tmp_path):
    rec = tr.build_episode(_synthetic_traj(), reward=0.0, task_id="t1")
    assert json.loads(tr.to_jsonl_line(rec)) == rec
    p = tmp_path / "episodes.jsonl"
    exp = tr.EpisodeExporter(p)
    r1 = exp.export(_synthetic_traj(), reward=1.0, task_id="t1", model_id="glm-5.2")
    r2 = exp.export(_synthetic_traj(), reward=0.0, task_id="t2", model_id="glm-5.2")
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["episode_id"] == r1["episode_id"]
    assert json.loads(lines[1])["outcome"]["reward"] == 0.0 and r2["task_id"] == "t2"


def test_accepts_raw_messages_list():
    rec = tr.build_episode(_synthetic_traj()["messages"], reward=1.0, task_id="t1")
    assert rec["n_turns"] == 5 and rec["outcome"]["reward"] == 1.0
    assert rec["outcome"]["exit_status"] == "Submitted"     # read from the trailing exit message
