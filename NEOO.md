# neooAgent — NeuraHash's SWE agent

**neooAgent** is NeuraHash's fork of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
(MIT — see `LICENSE.md` / `NOTICE`). The upstream agent is excellent and tiny (~100 lines, ~74% on
SWE-bench Verified); we keep it intact and add a thin **`neoo`** layer for our use case:

> the harness + RL **trajectory generator** for the GLM-5.2 benchmark assault
> (NeuraHash issues [#64](https://github.com/whitegreenstudios/neurahash/issues/64) umbrella,
> [#65](https://github.com/whitegreenstudios/neurahash/issues/65) repo-test reward,
> [#66](https://github.com/whitegreenstudios/neurahash/issues/66) agentic rollout,
> [#67](https://github.com/whitegreenstudios/neurahash/issues/67) this fork).

## Two integration points (this increment)

### 1. Point the agent at our served model
`src/minisweagent/neoo/model.py` reuses upstream's `LitellmModel` to hit our served GLM-5.2 over an
OpenAI-compatible endpoint:

```python
from minisweagent.neoo.model import neoo_model
model = neoo_model()   # NEOO_MODEL_NAME (default openai/glm-5.2), NEOO_API_BASE, NEOO_API_KEY
```

Requires the NeuraHash serve stack to speak `/v1/chat/completions` (or a thin shim in front).

### 2. Export episodes as verifiable-reward RL training data
`src/minisweagent/neoo/trajectory.py` turns each agent run into a content-addressed
`neoo-episode-1` record with the **terminal verifiable reward** attached (the repo-level test verdict
from #65), which the agentic RL rollout (#66) consumes:

```python
from minisweagent.neoo import EpisodeExporter
exp = EpisodeExporter("episodes.jsonl")
data = agent.save(None)                       # mini-swe-agent's serialized trajectory
reward = repo_tests_passed                    # 1.0 / 0.0 from the #65 checker (NOT the agent itself)
exp.export(data, reward=reward, task_id=instance_id, model_id="glm-5.2")
```

Each line is one episode: `{schema, episode_id (sha256), task_id, model_id, n_turns, turns[],
outcome{exit_status, submission, reward}, meta}`. Content-addressed for dedup + provenance, the same
discipline as NeuraHash's `data_collect` (#62).

## The flywheel

```
neooAgent solves SWE issues  →  episodes + verifiable reward (#65)  →  trajectory export
        ▲                                                                     │
        │                                                                     ▼
   better model  ◄─  verifiable-reward RL (pouw_rl #66, engines/ adapter)  ◄──┘
```

## Fork hygiene

- MIT retained (`LICENSE.md` + `NOTICE`).
- Upstream tracked via the `upstream` remote — `git fetch upstream && git merge upstream/main` to pull fixes.
- All NeuraHash code is additive under `src/minisweagent/neoo/`; the upstream agent loop is untouched.

## Honest scope

neooAgent is the **harness + trajectory generator** — necessary, not sufficient. The leaderboard
result needs the model + RL + data + compute (umbrella #64). The reward must come from the
**program checker** (#65), never the agent grading itself.
