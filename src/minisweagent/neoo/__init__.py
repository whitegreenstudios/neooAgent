"""
neoo — NeuraHash's integration layer over mini-swe-agent (this fork = neooAgent).

Adds, without touching the upstream agent loop:
  * trajectory export — turn agent episodes into verifiable-reward RL training data (trajectory.py)
  * a served-model helper — point the agent at NeuraHash's GLM-5.2 endpoint (model.py)

See NEOO.md. `neoo_model` lives in .model and is imported explicitly (it pulls in litellm), so this
package import stays cheap and dependency-light.
"""
from minisweagent.neoo.reward import extract_submission, submission_reward_fn
from minisweagent.neoo.run_episode import build_agent, run_episode, solve
from minisweagent.neoo.trajectory import (
    SCHEMA,
    EpisodeExporter,
    build_episode,
    episode_id,
    to_jsonl_line,
)

__all__ = [
    "SCHEMA",
    "EpisodeExporter",
    "build_episode",
    "episode_id",
    "to_jsonl_line",
    "run_episode",
    "build_agent",
    "solve",
    "submission_reward_fn",
    "extract_submission",
]
