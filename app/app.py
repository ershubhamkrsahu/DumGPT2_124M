import os
import random
import time
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
import gradio as gr
from model import load_dumgpt, GPTConfig

# ZeroGPU support for Hugging Face Spaces
try:
    import spaces
except ImportError:
    class MockSpaces:
        @staticmethod
        def GPU(func=None, duration=None):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func
    spaces = MockSpaces()

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running DumGPT Studio on {device}...")

# Model Registry
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "DumGPT-124M (Base - Trained From Scratch)": {
        "repo_id": "shubhamisok/DumGPT-124M",
        "stage": "Base Pretraining",
        "status": "Available",
        "badge": "🟢 Available",
        "description": (
            "124M-parameter decoder-only GPT-2 architecture trained from scratch on FineWeb-Edu. "
            "Reached cross-entropy loss ~6.48 at 1,000 steps."
        ),
        "link": "https://huggingface.co/shubhamisok/DumGPT-124M",
    },
    "DumGPT2-124M-RLVR (RL with Verifiable Rewards)": {
        "repo_id": "shubhamisok/DumGPT2-124M-RLVR",
        "stage": "RLVR Fine-Tune",
        "status": "Planned / In Development",
        "badge": "⏳ Coming Soon",
        "description": (
            "Post-training fine-tune applying Reinforcement Learning with Verifiable Rewards (RLVR) "
            "on the math-500 dataset for mathematical problem solving and reasoning."
        ),
        "link": "https://huggingface.co/shubhamisok/DumGPT2-124M-RLVR",
    },
    "DumGPT2-124M-Distill (Knowledge Distillation)": {
        "repo_id": "shubhamisok/DumGPT2-124M-Distill",
        "stage": "Distillation",
        "status": "Planned",
        "badge": "⏳ Coming Soon",
        "description": (
            "Knowledge distillation using a larger teacher model to compress reasoning "
            "and factual recall into the compact 124M parameter budget."
        ),
        "link": "https://huggingface.co/shubhamisok/DumGPT2-124M-Distill",
    },
}

# Cache for loaded models
MODELS_CACHE: Dict[str, Tuple[Any, Any, Any]] = {}

# Preload the default base model on startup
DEFAULT_MODEL_NAME = "DumGPT-124M (Base - Trained From Scratch)"
try:
    print(f"Preloading default model '{DEFAULT_MODEL_NAME}'...")
    base_model, base_enc, base_config = load_dumgpt("shubhamisok/DumGPT-124M", device=device)
    MODELS_CACHE["shubhamisok/DumGPT-124M"] = (base_model, base_enc, base_config)
    print("Default base model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not preload base model: {e}")


def get_model(model_name: str) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[str]]:
    info = AVAILABLE_MODELS.get(model_name)
    if not info:
        return None, None, None, f"Unknown model variant: {model_name}"

    repo_id = info["repo_id"]
    if repo_id in MODELS_CACHE:
        m, e, c = MODELS_CACHE[repo_id]
        return m, e, c, None

    # Attempt to load model dynamically
    try:
        print(f"Loading weights for '{model_name}' from {repo_id}...")
        m, e, c = load_dumgpt(repo_id, device=device)
        MODELS_CACHE[repo_id] = (m, e, c)
        return m, e, c, None
    except Exception as exc:
        msg = (
            f"⚠️ **{model_name}** is not yet available for inference on Hugging Face.\n\n"
            f"**Repository**: [{repo_id}](https://huggingface.co/{repo_id})\n"
            f"**Status**: {info['badge']}\n"
            f"**Details**: {info['description']}\n\n"
            f"*As soon as weights (`pytorch_model.bin`) are published to this repo, "
            f"the studio will automatically load and serve it without requiring code updates.*"
        )
        return None, None, None, msg


def update_model_info(model_name: str) -> str:
    info = AVAILABLE_MODELS.get(model_name, {})
    badge = info.get("badge", "")
    desc = info.get("description", "")
    link = info.get("link", "#")
    return f"**Model Status**: {badge} &nbsp;|&nbsp; [Hugging Face Repository]({link})\n\n{desc}"


@spaces.GPU(duration=60)
def generate_stream(
    model_name: str,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 42,
):
    if not prompt or not prompt.strip():
        yield "Please provide a prompt to start generation."
        return

    model_inst, tokenizer_inst, config_inst, err_msg = get_model(model_name)
    if err_msg is not None:
        yield err_msg
        return

    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_inst = model_inst.to(run_device)

    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
        random.seed(seed)

    tokens = tokenizer_inst.encode(prompt)
    if len(tokens) > config_inst.block_size - 10:
        tokens = tokens[-(config_inst.block_size - 10):]

    x = torch.tensor(tokens, dtype=torch.long, device=run_device).unsqueeze(0)
    generated_text = prompt
    start_time = time.time()

    yield generated_text

    with torch.no_grad():
        for i in range(int(max_new_tokens)):
            x_cond = x[:, -config_inst.block_size:]
            logits = model_inst(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)

            if top_k > 0:
                topk_probs, topk_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                probs = F.softmax(topk_probs, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                next_token = topk_indices.gather(-1, next_idx)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_token), dim=1)
            new_char = tokenizer_inst.decode([next_token.item()])
            generated_text += new_char

            if i % 2 == 0 or i == max_new_tokens - 1:
                yield generated_text

    dt = time.time() - start_time
    tok_sec = max_new_tokens / max(dt, 1e-4)
    summary = (
        f"\n\n---\n"
        f"*{model_name} • Generated {max_new_tokens} tokens in {dt:.2f}s "
        f"({tok_sec:.1f} tokens/sec on {run_device.upper()})*"
    )
    yield generated_text + summary


# Custom theme styling
custom_css = """
.container { max-width: 960px; margin: auto; }
.output-box textarea { font-family: 'Courier New', monospace; font-size: 15px; line-height: 1.4; }
.info-card { background: #f8fafc; padding: 12px 16px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 12px; }
"""

with gr.Blocks(title="DumGPT Studio", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 DumGPT Studio
        **Interactive multi-model playground for the DumGPT family: Base, RLVR, and Distilled models.**
        
        *HF Space:* [`shubhamisok/dumgpt-studio`](https://huggingface.co/spaces/shubhamisok/dumgpt-studio) • 
        *Source Code:* [`GitHub - DumGPT2_124M`](https://github.com/ShubhamKumarSahu-svg/DumGPT2_124M)
        """
    )

    with gr.Tabs():
        with gr.TabItem("🚀 Playground"):
            with gr.Row():
                with gr.Column(scale=3):
                    model_selector = gr.Dropdown(
                        choices=list(AVAILABLE_MODELS.keys()),
                        value=DEFAULT_MODEL_NAME,
                        label="Select DumGPT Model Variant",
                        interactive=True,
                    )
                    model_info_box = gr.Markdown(
                        value=update_model_info(DEFAULT_MODEL_NAME),
                        elem_classes=["info-card"],
                    )
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Type a sentence starter (e.g. 'The history of artificial intelligence begins with')...",
                        lines=3,
                        value="The history of artificial intelligence begins with",
                    )
                    with gr.Row():
                        generate_btn = gr.Button("Generate", variant="primary", scale=2)
                        clear_btn = gr.Button("Clear", scale=1)

                    output_text = gr.Textbox(
                        label="DumGPT Output (Streaming)",
                        lines=8,
                        elem_classes=["output-box"],
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Generation Settings")
                    max_tokens_slider = gr.Slider(
                        minimum=10, maximum=256, value=60, step=10, label="Max New Tokens"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1, maximum=1.8, value=0.8, step=0.05, label="Temperature"
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=100, value=50, step=5, label="Top-K Sampling"
                    )
                    seed_input = gr.Number(
                        value=42, label="Random Seed", precision=0
                    )

            gr.Examples(
                examples=[
                    [DEFAULT_MODEL_NAME, "The history of artificial intelligence begins with", 60, 0.8, 50, 42],
                    [DEFAULT_MODEL_NAME, "Once upon a time, in a land far away,", 60, 0.8, 50, 1337],
                    [DEFAULT_MODEL_NAME, "The fundamental law of nature states that", 70, 0.75, 40, 2024],
                    [DEFAULT_MODEL_NAME, "In modern computer science, algorithms are", 60, 0.8, 50, 777],
                ],
                inputs=[model_selector, prompt_input, max_tokens_slider, temperature_slider, top_k_slider, seed_input],
                label="Sample Prompts",
            )

            # Event listeners
            model_selector.change(
                fn=update_model_info,
                inputs=model_selector,
                outputs=model_info_box,
            )

            generate_btn.click(
                fn=generate_stream,
                inputs=[model_selector, prompt_input, max_tokens_slider, temperature_slider, top_k_slider, seed_input],
                outputs=output_text,
            )
            clear_btn.click(lambda: ("", ""), outputs=[prompt_input, output_text])

        with gr.TabItem("📊 Architecture & Roadmap"):
            gr.Markdown(
                r"""
                ### DumGPT Model Suite Roadmap
                
                | Model | Stage | Status | Details |
                | :--- | :--- | :--- | :--- |
                | **DumGPT-124M** | Base | **Available** | Trained from scratch on FineWeb-Edu (1k steps, loss 6.45). |
                | **DumGPT2-124M-RLVR** | Fine-tune | *In Development* | Reinforcement learning with verifiable rewards on `math-500`. |
                | **DumGPT2-124M-Distill** | Distilled | *Planned* | Knowledge distillation from a larger teacher model. |
                
                ---
                
                ### Base Model Specifications
                - **Architecture**: GPT-2 (Transformer Decoder-only)
                - **Parameters**: 124,475,904 (~124M)
                - **Layers**: 12
                - **Attention Heads**: 12
                - **Embedding Dimension**: 768
                - **Context Window**: 1,024 tokens
                - **Vocabulary Size**: 50,304 (padded to multiple of 128 for tensor core efficiency)
                - **Weight Tying**: Token embedding `wte` and `lm_head` shared
                
                ### Training Highlights
                - **Pretrained From Scratch**: Initialized from Gaussian noise with initial cross-entropy loss ~10.8 ($\ln(50,304)$).
                - **Dataset**: `HuggingFaceFW/fineweb-edu` (sample-10BT), tokenized with `tiktoken` into binary `uint16` shards.
                - **Optimization**: AdamW with weight decay 0.1, warmup to peak LR $6 \times 10^{-4}$ with cosine decay to $6 \times 10^{-5}$, gradient accumulation.
                - **Precision**: Mixed precision (BF16 / FP16) with Flash Attention (SDPA).
                - **Loss**: Training loss reached ~6.48 and validation loss reached 6.4468 at 1,000 steps.
                """
            )

        with gr.TabItem("📈 Training Curves & Metrics"):
            gr.Markdown(
                """
                ### Base Model Training Plots
                All curves were logged and generated during the 1,000-step pretraining run:
                
                | Metric | Visual Plot |
                | :--- | :--- |
                | **Training Loss Curve** | [View train_loss.png](https://huggingface.co/shubhamisok/DumGPT-124M/resolve/main/outputs/plots/train_loss.png) |
                | **Validation Loss Curve** | [View val_loss.png](https://huggingface.co/shubhamisok/DumGPT-124M/resolve/main/outputs/plots/val_loss.png) |
                | **Learning Rate Schedule** | [View learning_rate.png](https://huggingface.co/shubhamisok/DumGPT-124M/resolve/main/outputs/plots/learning_rate.png) |
                | **Gradient Norm** | [View grad_norm.png](https://huggingface.co/shubhamisok/DumGPT-124M/resolve/main/outputs/plots/grad_norm.png) |
                | **Tokens / Second** | [View tokens_per_sec.png](https://huggingface.co/shubhamisok/DumGPT-124M/resolve/main/outputs/plots/tokens_per_sec.png) |
                
                *(You can explore all raw logs in the [Base Model Repository](https://huggingface.co/shubhamisok/DumGPT-124M/tree/main/outputs).)*
                """
            )

if __name__ == "__main__":
    demo.queue().launch()
