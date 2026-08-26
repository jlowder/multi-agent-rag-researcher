"""Phase 5: output_format defaults are JSON; standard mode is pinned to Markdown."""

import importlib
import inspect

import deep_research_orchestrator as dpo
import orchestrator_agent as oagent
wmod = importlib.import_module("worker_agents.writer_agent")


def test_writer_function_defaults_are_json():
    for fn in (wmod.writer_agent, wmod.write_section, wmod.write_synthesis):
        default = inspect.signature(fn).parameters["output_format"].default
        assert default == "json", f"{fn.__name__} defaults to {default!r}"


def test_deep_research_default_is_json():
    default = inspect.signature(dpo.deep_research).parameters["output_format"].default
    assert default == "json"


def test_standard_orchestrator_pins_markdown():
    """The standard-mode writer call must pass output_format="markdown"
    explicitly, because the writer default flipped to JSON (Phase 5)."""
    src = inspect.getsource(oagent)
    assert 'output = writer_agent(' in src
    # The call block must carry the explicit markdown pin.
    idx = src.index("output = writer_agent(")
    block = src[idx : idx + 400]
    assert 'output_format="markdown"' in block


def test_ui_save_handler_prefers_structured():
    src = inspect.getsource(importlib.import_module("ui.gradio_handlers"))
    assert 'report_state.get("report_json")' in src
    assert "save_structured_report" in src
