"""
neoo.trajectory — turn a neooAgent (mini-swe-agent) episode into verifiable-reward RL training data.

mini-swe-agent's DefaultAgent already SAVES a full trajectory (messages + info) via `agent.save()`,
but for NeuraHash's RL flywheel we need three more things:

  1. a TERMINAL VERIFIABLE REWARD attached to the episode — the discrete program verdict from the
     repo-level test-execution checker (NeuraHash issue #65: did the patch make the hidden tests
     pass?). mini-swe-agent's trajectory has exit_status/submission but NOT a reward.
  2. a COMPACT, STABLE schema (`neoo-episode-1`) the RL trainer can consume directly — the agentic
     multi-turn rollout (#66) reads (turns -> terminal reward) episodes.
  3. CONTENT-ADDRESSING — a sha256 episode_id over the canonical record, so episodes dedup and carry
     provenance the same way data_collect (#62) handles task shards.

Deliberately DEPENDENCY-FREE (stdlib only, no `minisweagent` imports) so it is testable without the
agent's runtime deps and can be reused by the NeuraHash training side.
"""

import hashlib
import json
from pathlib import Path

SCHEMA = "neoo-episode-1"


def _messages_and_info(data):
    """Accept either DefaultAgent.serialize() output ({"messages", "info", ...}) or a raw messages
    list. Returns (messages, info)."""
    if isinstance(data, dict):
        return data.get("messages", []) or [], data.get("info", {}) or {}
    return list(data), {}


def _turn(msg):
    """Compact a mini-swe-agent message into a training turn: role + content (+ actions if any)."""
    extra = (msg.get("extra") or {}) if isinstance(msg, dict) else {}
    turn = {"role": msg.get("role", ""), "content": msg.get("content", "")}
    if extra.get("actions"):
        turn["actions"] = extra["actions"]
    return turn


def episode_id(record):
    """Content address of an episode: sha256 over the canonical record minus the id field itself."""
    body = {k: v for k, v in record.items() if k != "episode_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_episode(data, *, reward, task_id, model_id=""):
    """Normalize a neooAgent run into a content-addressed `neoo-episode-1` record.

    `data`     — DefaultAgent.serialize() dict or a raw messages list
    `reward`   — the terminal verifiable reward (the #65 repo-test verdict), e.g. 1.0/0.0
    `task_id`  — provenance: which SWE instance this episode is for (dedup + decontam key)
    `model_id` — which model produced it
    """
    messages, info = _messages_and_info(data)
    turns = [_turn(m) for m in messages if isinstance(m, dict) and m.get("role") != "exit"]
    last = messages[-1] if messages else {}
    last_extra = (last.get("extra") or {}) if isinstance(last, dict) else {}
    stats = info.get("model_stats") or {}
    record = {
        "schema": SCHEMA,
        "episode_id": "",
        "task_id": task_id,
        "model_id": model_id,
        "n_turns": len(turns),
        "turns": turns,
        "outcome": {
            "exit_status": info.get("exit_status", last_extra.get("exit_status", "")),
            "submission": info.get("submission", last_extra.get("submission", "")),
            "reward": float(reward),
        },
        "meta": {
            "cost": stats.get("instance_cost", 0.0),
            "api_calls": stats.get("api_calls", 0),
            "trajectory_format": data.get("trajectory_format") if isinstance(data, dict) else None,
        },
    }
    record["episode_id"] = episode_id(record)
    return record


def to_jsonl_line(record):
    """One canonical JSONL line (sorted keys) — stable across machines for content-addressing."""
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


class EpisodeExporter:
    """Append neooAgent episodes to a JSONL file as verifiable-reward RL training data.

        exp = EpisodeExporter("episodes.jsonl")
        exp.export(agent.save(None), reward=repo_tests_passed, task_id=instance_id, model_id="glm-5.2")
    """

    def __init__(self, path):
        self.path = Path(path)

    def export(self, data, *, reward, task_id, model_id=""):
        rec = build_episode(data, reward=reward, task_id=task_id, model_id=model_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(to_jsonl_line(rec) + "\n")
        return rec
