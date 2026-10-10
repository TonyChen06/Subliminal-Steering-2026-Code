"""Alternate fixed teachers after each delimited number, sharing sampled text."""
import re
from contextlib import contextmanager

import torch


def completed_numbers(text):
    # A delimiter, not a digit token, marks the end of a number.
    return len(re.findall(r"\d+(?=\D)", text))


@contextmanager
def alternating_teachers(model, tokenizer, hook, first, second):
    """Evaluate both fixed teachers with independent KV-cache batch rows.

    Hugging Face still performs the original sampling; only the logits' source
    changes. Rows alternate which teacher starts. Both caches receive every
    sampled token, including tokens chosen by the other teacher.
    """
    forward = model.forward
    texts = None
    traces = []

    def alternating_forward(input_ids=None, attention_mask=None, **kwargs):
        nonlocal texts
        batch = input_ids.shape[0]
        if texts is None:
            texts = [""] * batch
        else:
            pieces = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            texts = [text + piece for text, piece in zip(texts, pieces)]
        choices = [(i + completed_numbers(text)) % 2 for i, text in enumerate(texts)]
        traces.append(choices)
        hook.sv = torch.stack([first] * batch + [second] * batch).unsqueeze(1)
        expanded = dict(kwargs)
        if expanded.get("position_ids") is not None:
            expanded["position_ids"] = expanded["position_ids"].repeat(2, 1)
        output = forward(
            input_ids=input_ids.repeat(2, 1),
            attention_mask=attention_mask.repeat(2, 1) if attention_mask is not None else None,
            **expanded,
        )
        rows = torch.arange(batch, device=output.logits.device)
        rows += torch.tensor(choices, device=rows.device) * batch
        output.logits = output.logits[rows]
        return output

    model.forward = alternating_forward
    try:
        yield traces
    finally:
        model.forward = forward
        hook.sv = first
