"""Inference: everything about HOW ParlorWave talks to a language model.

The ParlorWave core decides WHAT to ask and owns every fact, rule and decision; this
package owns the runtime transport, model selection, and (in later phases)
prompt-profile rendering, chat template, sampling, token counting and
context-window enforcement.

Dependency direction is one-way and enforced by test:

    core         ->  inference        allowed
    inference    ->  core             FORBIDDEN
"""
