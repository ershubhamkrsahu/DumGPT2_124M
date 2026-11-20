import secrets
import time
from typing import Any

import gradio as gr
import torch
import torch.nn.functional as F

from model import load_dumgpt

try:
    import spaces
except ImportError:
    class MockSpaces:
        @staticmethod
        def GPU(func=None, duration=None):
            if func is None:
                return lambda wrapped: wrapped
            return func

    spaces = MockSpaces()


MODEL_REPO = "shubhamisok/DumGPT-124M"
MODEL_FILENAME = "DumGPT-124M-base.pth"
DEFAULT_MAX_TOKENS = 80
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
RANDOM_SEED = -1

MODEL_CACHE: dict[str, tuple[Any, Any, Any]] = {}



def get_model(device: str):
    cache_key = f"{MODEL_REPO}:{MODEL_FILENAME}:{device}"
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = load_dumgpt(
            MODEL_REPO,
            filename=MODEL_FILENAME,
            device=device,
        )
    return MODEL_CACHE[cache_key]


@spaces.GPU(duration=60)
def stream_completion(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, tokenizer, config = get_model(device)
    except Exception as exc:
        yield (
            f"The model could not load. Try again in a moment.\n\n`{exc}`",
            "Model load failed",
        )
        return

    sampling_seed = secrets.randbits(63) if seed in (None, RANDOM_SEED) else int(seed)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)

    prompt_tokens = tokenizer.encode(prompt)
    available_context = max(1, config.block_size - int(max_new_tokens))
    was_trimmed = len(prompt_tokens) > available_context
    prompt_tokens = prompt_tokens[-available_context:]

    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    continuation = ""
    started_at = time.perf_counter()

    with torch.inference_mode():
        for index in range(int(max_new_tokens)):
            context = tokens[:, -config.block_size:]
            logits = model(context)[:, -1, :] / max(float(temperature), 1e-4)

            if int(top_k) > 0:
                values, indices = torch.topk(logits, min(int(top_k), logits.size(-1)), dim=-1)
                next_token = indices.gather(
                    -1,
                    torch.multinomial(F.softmax(values, dim=-1), num_samples=1, generator=generator),
                )
            else:
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=generator)

            tokens = torch.cat((tokens, next_token), dim=1)
            continuation += tokenizer.decode([next_token.item()])
            if index % 3 == 0 or index == int(max_new_tokens) - 1:
                yield continuation, f"Writing · {index + 1} of {int(max_new_tokens)} tokens"

    elapsed = time.perf_counter() - started_at
    speed = int(max_new_tokens) / max(elapsed, 1e-4)
    trim_note = " · prompt trimmed to fit context" if was_trimmed else ""
    yield continuation, f"Finished · {speed:.1f} tokens/sec on {device.upper()}{trim_note}"


def respond(message, history, max_tokens, temperature, top_k, seed):
    if not message or not message.strip():
        raise gr.Error("Enter a sentence to continue.")
    for continuation, _ in stream_completion(message, max_tokens, temperature, top_k, seed):
        yield continuation


# Build outside Blocks so ChatInterface places these in its collapsed accordion.
settings = [
    gr.Slider(20, 256, value=DEFAULT_MAX_TOKENS, step=4, label="New tokens"),
    gr.Slider(0.1, 1.5, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature"),
    gr.Slider(1, 100, value=DEFAULT_TOP_K, step=1, label="Top-k"),
    gr.Number(value=RANDOM_SEED, minimum=-1, precision=0, label="Seed (−1 for random)"),
]

with gr.Blocks(title="DumGPT Studio") as demo:
    gr.Markdown("# DumGPT Studio")
    with gr.Tab("Base"):
        gr.Markdown("### DumGPT-124M Base\nStart a sentence and the model will continue it. Each prompt is generated independently.")
        gr.ChatInterface(
            respond,
            type="messages",
            additional_inputs=settings,
            additional_inputs_accordion=gr.Accordion("Generation settings", open=False, render=False),
            fill_height=False,
            api_name=False,
        )
    with gr.Tab("RLVR"):
        gr.Markdown("### DumGPT2-124M-RLVR\n**In development**\n\nThis model will use verifiable rewards to train on mathematical reasoning problems.")
    with gr.Tab("Distill"):
        gr.Markdown("### DumGPT2-124M-Distill\n**Planned**\n\nThis experiment will train a 124M-parameter student using a larger teacher model.")


if __name__ == "__main__":
    demo.queue().launch()
