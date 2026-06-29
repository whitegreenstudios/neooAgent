"""
neoo.model — point neooAgent at NeuraHash's served model (e.g. GLM-5.2) over an OpenAI-compatible
endpoint, reusing mini-swe-agent's existing LitellmModel (no new model client needed).

    from minisweagent.neoo.model import neoo_model
    model = neoo_model()            # reads NEOO_MODEL_NAME / NEOO_API_BASE / NEOO_API_KEY

Requires the NeuraHash serve endpoint to speak the OpenAI Chat Completions API
(/v1/chat/completions). litellm's `openai/<name>` provider routes there via api_base. If our serve
stack isn't OpenAI-compatible yet, put a thin shim in front of it (tracked in NeuraHash #67).

litellm is imported lazily (inside the function), so importing this module costs nothing until used.
"""

import os


def neoo_model(**overrides):
    """Build a LitellmModel pointed at our served model. Env:
      NEOO_MODEL_NAME  (default 'openai/glm-5.2')
      NEOO_API_BASE    (our serve endpoint URL)
      NEOO_API_KEY     (token-gate key, if any)
    `overrides` are merged into the LitellmModel config (e.g. model_kwargs)."""
    from minisweagent.models.litellm_model import LitellmModel

    model_kwargs = dict(overrides.pop("model_kwargs", {}))
    if os.getenv("NEOO_API_BASE"):
        model_kwargs.setdefault("api_base", os.environ["NEOO_API_BASE"])
    if os.getenv("NEOO_API_KEY"):
        model_kwargs.setdefault("api_key", os.environ["NEOO_API_KEY"])
    cfg = {
        "model_name": os.getenv("NEOO_MODEL_NAME", "openai/glm-5.2"),
        "model_kwargs": model_kwargs,
    }
    cfg.update(overrides)
    return LitellmModel(**cfg)
