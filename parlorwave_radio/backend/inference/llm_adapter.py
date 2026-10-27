"""The inference seam: the ParlorWave core asks a ROLE for text, not a model for a
completion.

PASS-THROUGH ONLY. This module is dependency inversion and nothing
else. It sends byte-for-byte the prompt it is given, to the same endpoint, with
the same payload fields, the same model resolution and the same timeout the DJ
writer used before it existed, and it propagates failures exactly as
dj_engine.default_llm did. Nothing a listener can hear changes.

Deliberately NOT here yet (each is its own later change, because each can change
what the listener hears):

  * prompt-profile rendering, section ordering, chat-template selection
  * explicit sampling (temperature/top_p/stop) -- today's behaviour is whatever
    the installed model's Modelfile says, and making it explicit is a separate change
  * token counting and context-window enforcement
  * the ok/error result mapping of ModelContractV1 (see below)
  * model packages, staged slots, certification

Failure semantics, verbatim from what it replaces: providers raises on
network/HTTP/JSON failure and those exceptions PROPAGATE; a missing 'response'
field degrades to '' (render_break turns empty text into a safe failed break).
The ModelContractV1 {"ok": false, "error": ...} envelope is deliberately NOT
implemented here -- returning a result object where the old code raised would
change failure semantics, which a pass-through must not do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import providers
from inference import model_roles

# --- model roles --------------------------------------------------------------
# Three distinct JOBS, not three copies of one model knob. Roles are routed,
# certified and updated independently: moving dj_writer to a new package must
# never silently change the other two. The literals live in ONE place
# (inference.model_roles); these names are re-exports for the call sites.
ROLE_DJ_WRITER = model_roles.ROLE_DJ_WRITER
ROLE_STATION_PROGRAMMER = model_roles.ROLE_STATION_PROGRAMMER
ROLE_SONG_CONTEXT_EXTRACTOR = model_roles.ROLE_SONG_CONTEXT_EXTRACTOR
ROLES = model_roles.ROLES

# dj_writer resolution, moved here VERBATIM from dj_engine: same env variables,
# same fallback to the shared station default, same 60s budget. dj_engine keeps
# DJ_LLM_MODEL / DJ_LLM_TIMEOUT_S as aliases of these, so there is exactly one
# source of truth and the long-standing env contract is untouched.
DJ_WRITER_MODEL = os.environ.get("DJ_LLM_MODEL", providers.OLLAMA_MODEL)
DJ_WRITER_TIMEOUT_S = float(os.environ.get("DJ_LLM_TIMEOUT_S", "60"))

# station_programmer and song_context_extractor are NOT resolved here yet.
# They keep their own resolution (settings key and module constant); the roles
# are named here so the registry is honest about what exists, and so a later
# change moves them one at a time rather than inventing a global model knob.


@dataclass(frozen=True)
class TextRequest:
    """A request for free text from a role.

    `prompt` is the FINAL prompt string. The adapter builds no prompt:
    Core still builds the prompt and the adapter forwards its bytes unchanged.
    """

    role: str
    prompt: str


@dataclass(frozen=True)
class TextResult:
    """Text from a role, with the identity that produced it.

    Typed rather than a bare str so ModelContractV1 can add usage, package
    identity and the ok/error envelope later without touching call sites.
    """

    text: str
    role: str
    model: str


class OllamaTextAdapter:
    """Free-text generation over the current Ollama runtime.

    One class per runtime. Core never names this class -- it asks for the
    configured adapter -- so llama.cpp or another compatible engine is a
    sibling, not a rewrite.
    """

    engine = "ollama"

    def model_for(self, role: str) -> str:
        if role == ROLE_DJ_WRITER:
            return DJ_WRITER_MODEL
        raise ValueError(f"role not served by this adapter: {role!r}")

    def timeout_for(self, role: str) -> float:
        if role == ROLE_DJ_WRITER:
            return DJ_WRITER_TIMEOUT_S
        raise ValueError(f"role not served by this adapter: {role!r}")

    def serves(self, role: str) -> bool:
        return role == ROLE_DJ_WRITER

    def endpoint(self) -> tuple[str, str]:
        """(generate url, key) for the writing model, resolved AT CALL TIME.

        Stored settings win over the environment, for the same reason the
        voice engine's do: the address was an environment variable read
        once at import, so changing it meant restarting the add-on, and
        nothing in the product could show what it was.

        The url is the GENERATE ENDPOINT, not the address as typed. This
        used to hand `providers.ollama_generate` the raw settings
        base_url, which POSTs to Ollama's root and is answered 405 -- so
        the first installation to configure its model on the screen
        instead of in the environment would have had every break fail
        over to deterministic copy. `optional_services.effective_writer`
        is the one place that completes the address.

        Resolved AT CALL TIME and cached nowhere, which is what makes a
        changed model take effect at the next break rather than at the
        next add-on restart."""
        try:
            import optional_services
            cfg = optional_services.effective_writer()
            if cfg["usable"]:
                return cfg["endpoint"], cfg["api_key"]
        except Exception:  # noqa: BLE001 - never break the writer
            pass
        return "", ""

    def model_override(self) -> str:
        """The model the owner selected, or '' to use the env contract."""
        try:
            import optional_services
            cfg = optional_services.effective_writer()
            if cfg["usable"] and cfg.get("model"):
                return str(cfg["model"])
        except Exception:  # noqa: BLE001
            pass
        return ""

    def generate_text(self, request: TextRequest) -> TextResult:
        """The prompt's bytes, to the runtime, and back.

        No `format`, no `raw`, no sampling options: the payload is the three
        fields the DJ writer has always sent plus ONE runtime option, the
        writer's context window (WRITER_NUM_CTX). A missing 'response'
        degrades to ''; every other failure propagates.
        """
        model = self.model_override() or self.model_for(request.role)
        url, key = self.endpoint()
        data = providers.ollama_generate(
            request.prompt, model, timeout=self.timeout_for(request.role),
            url=url or None, api_key=key or None, options=WRITER_OPTIONS)
        return TextResult(text=str(data.get("response") or "").strip(),
                          role=request.role, model=model)

    def warm(self, role: str) -> bool:
        """Best-effort warm-on-intent. Never raises: a cold model is slower,
        not broken."""
        try:
            providers.ollama_warm(self.model_for(role),
                                  timeout=self.timeout_for(role),
                                  options=WRITER_OPTIONS)
            return True
        except Exception:  # noqa: BLE001 - warming is never load-bearing
            return False


#: The DJ writer's context window, the same on every call that reaches the
#: writer model: the Smart planner and story preparation send
#: smart_dj.PLANNER_NUM_CTX, and a call with any other window -- or none,
#: which Ollama 0.40 loads at the model's full 131072 -- makes the runtime
#: reload the model (seconds of load_duration on a free-text call after
#: planner calls, and the next planner call reloads back). The free-text
#: prompts fit with the planner's own margin (twice the
#: prompt plus the answer; test_llm_adapter.WriterWindowTests).
WRITER_NUM_CTX = 6144
WRITER_OPTIONS = {"num_ctx": WRITER_NUM_CTX}

_DEFAULT = OllamaTextAdapter()


def default_adapter() -> OllamaTextAdapter:
    """The configured adapter. A module-level function (not a bare constant)
    so a later change can select per role from a model package without any call
    site changing."""
    return _DEFAULT


def generate_text(role: str, prompt: str) -> str:
    """Convenience for the one thing Core does today: text from a role."""
    return default_adapter().generate_text(
        TextRequest(role=role, prompt=prompt)).text
