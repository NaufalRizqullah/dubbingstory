"""Chronological request groups sized by the actual local text tokenizer."""

from dubbingstory.llm.text_client import ContextBudgetError


def context_batching_enabled(cfg) -> bool:
    return bool(getattr(cfg, "llm_batching_enabled", False))


def budgeted_groups(items, prompt_builder, client, *, task_name="text", max_items=4,
                    max_output_tokens=None):
    """Yield every item once, splitting only requests that exceed capacity.

    Prompts are built immediately before each yield, so later groups see any
    validated continuity state collected by the caller after previous yields.
    A missing preflight method (legacy/custom adapters) keeps bounded grouping.
    A single oversized item raises; neither evidence nor output is truncated.
    """
    if type(max_items) is not int or max_items <= 0:
        raise ValueError("max_items must be a positive integer")
    ordered = list(items)
    preflight = getattr(client, "preflight", None)

    def fit(group):
        if callable(preflight):
            try:
                preflight(prompt_builder(group), task_name=task_name,
                          max_output_tokens=max_output_tokens)
            except ContextBudgetError:
                if len(group) == 1:
                    raise
                middle = len(group) // 2
                yield from fit(group[:middle])
                yield from fit(group[middle:])
                return
        yield group

    for start in range(0, len(ordered), max_items):
        yield from fit(ordered[start:start + max_items])
