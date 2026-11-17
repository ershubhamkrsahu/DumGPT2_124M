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


# ---------------------------------------------------------------------------
# UI: theme, styling and layout (presentation only)
# ---------------------------------------------------------------------------

# Palette
#   Ink        #0b2239   hero band, strong text
#   Steel teal #0f7a98   primary accent (buttons, focus, links)
#   Canvas     #f3f5f8   page background
#   Surface    #ffffff   cards and inputs
#   Border     #d8dfe7   hairlines
#   Muted      #5a6877   secondary text
STEEL_TEAL = gr.themes.Color(
    c50="#ecfbfd",
    c100="#d0f3f8",
    c200="#a5e6f0",
    c300="#6dd2e4",
    c400="#33b6d0",
    c500="#1597b5",
    c600="#0f7a98",
    c700="#0e6179",
    c800="#104f62",
    c900="#124253",
    c950="#082a37",
    name="steel_teal",
)

custom_theme = gr.themes.Soft(
    primary_hue=STEEL_TEAL,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    # Page and surfaces
    body_background_fill="#f3f5f8",
    body_background_fill_dark="#0a1119",
    body_text_color="#17212b",
    body_text_color_dark="#e6edf3",
    body_text_color_subdued="#5a6877",
    body_text_color_subdued_dark="#92a1b0",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#111b26",
    background_fill_secondary="#f7f9fb",
    background_fill_secondary_dark="#0d1620",
    block_background_fill="#ffffff",
    block_background_fill_dark="#111b26",
    block_border_width="1px",
    block_border_color="#d8dfe7",
    block_border_color_dark="#223142",
    block_shadow="0 1px 2px rgba(11, 34, 57, 0.05)",
    block_label_text_weight="600",
    block_title_text_weight="600",
    border_color_primary="#d8dfe7",
    border_color_primary_dark="#223142",
    # Inputs
    input_background_fill="#f9fafc",
    input_background_fill_dark="#0d1620",
    input_border_color="#d8dfe7",
    input_border_color_dark="#223142",
    input_border_color_focus="#0f7a98",
    input_border_color_focus_dark="#5ec4d8",
    slider_color="#0f7a98",
    slider_color_dark="#5ec4d8",
    # Buttons
    button_primary_background_fill="#0e6179",
    button_primary_background_fill_hover="#0b4f64",
    button_primary_background_fill_dark="#0f7a98",
    button_primary_background_fill_hover_dark="#1597b5",
    button_primary_border_color="#0e6179",
    button_primary_border_color_dark="#0f7a98",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#eef2f6",
    button_secondary_background_fill_dark="#111b26",
    button_secondary_background_fill_hover_dark="#18273a",
    button_secondary_border_color="#d8dfe7",
    button_secondary_border_color_dark="#223142",
    button_secondary_text_color="#17212b",
    button_secondary_text_color_dark="#e6edf3",
    # Links
    link_text_color="#0e6179",
    link_text_color_dark="#5ec4d8",
    link_text_color_hover="#0b4f64",
    link_text_color_hover_dark="#8ad8e7",
    link_text_color_active="#0b4f64",
    link_text_color_active_dark="#8ad8e7",
)

custom_css = """
:root {
    --ds-ink: #0b2239;
    --ds-accent: #0f7a98;
    --ds-accent-strong: #0e6179;
    --ds-canvas: #f3f5f8;
    --ds-surface: #ffffff;
    --ds-surface-2: #f7f9fb;
    --ds-border: #d8dfe7;
    --ds-text: #17212b;
    --ds-muted: #5a6877;
}
.dark {
    --ds-ink: #0b2239;
    --ds-accent: #5ec4d8;
    --ds-accent-strong: #8ad8e7;
    --ds-canvas: #0a1119;
    --ds-surface: #111b26;
    --ds-surface-2: #0d1620;
    --ds-border: #223142;
    --ds-text: #e6edf3;
    --ds-muted: #92a1b0;
}

/* Page container */
.gradio-container {
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding: 28px 20px 12px !important;
}
.container { max-width: 960px; margin: auto; }

/* Hero */
.ds-hero-wrap {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.ds-hero {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
    gap: 32px;
    align-items: center;
    padding: 34px 38px;
    border-radius: 14px;
    background: var(--ds-ink);
    box-shadow: inset 0 -3px 0 #1597b5;
}
.ds-hero h1 {
    margin: 0 0 10px !important;
    color: #ffffff !important;
    font-size: 2.35rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    line-height: 1.12 !important;
    border: none !important;
}
.ds-hero p.ds-lede {
    margin: 0 0 22px !important;
    max-width: 46ch;
    color: #b6c7d8 !important;
    font-size: 1.02rem;
    line-height: 1.6;
}
.ds-hero-links { display: flex; flex-wrap: wrap; gap: 10px; }
.ds-hero a.ds-link {
    display: inline-flex;
    align-items: center;
    padding: 9px 16px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.28);
    color: #ffffff !important;
    font-size: 0.92rem;
    font-weight: 500;
    text-decoration: none !important;
    transition: background-color 0.15s ease;
}
.ds-hero a.ds-link:hover { background: rgba(255, 255, 255, 0.1); }
.ds-hero a.ds-link.ds-link-solid {
    background: #ffffff;
    border-color: #ffffff;
    color: var(--ds-ink) !important;
}
.ds-hero a.ds-link.ds-link-solid:hover { background: #e4edf4; }

.ds-spec {
    padding: 18px 22px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.14);
}
.ds-spec-title {
    margin: 0 0 8px;
    color: #ffffff;
    font-size: 0.95rem;
    font-weight: 600;
}
.ds-spec dl { margin: 0; }
.ds-spec .ds-row {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    font-size: 0.9rem;
}
.ds-spec .ds-row:last-child { border-bottom: none; }
.ds-spec dt { color: #9db4c8; margin: 0; }
.ds-spec dd {
    margin: 0;
    color: #ffffff;
    font-weight: 500;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* Tabs */
button[role="tab"],
.tab-nav button {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    color: var(--ds-muted);
}
button[role="tab"][aria-selected="true"],
.tab-nav button.selected {
    color: var(--ds-accent) !important;
    font-weight: 600 !important;
}

/* Model info card */
.info-card {
    background: var(--ds-surface-2) !important;
    border: 1px solid var(--ds-border) !important;
    border-left: 3px solid var(--ds-accent) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    margin-bottom: 12px;
}
.info-card p {
    margin: 0 0 0.45em;
    color: var(--ds-text);
    line-height: 1.55;
}
.info-card p:last-child { margin-bottom: 0; }
.info-card a { color: var(--ds-accent); font-weight: 500; }

/* Prompt and output */
.output-box textarea {
    font-family: "IBM Plex Mono", ui-monospace, "SF Mono", Consolas, monospace !important;
    font-size: 14.5px;
    line-height: 1.65;
}

/* Buttons */
button.primary, button.secondary { min-height: 42px; font-weight: 600; }

/* Settings panel */
.ds-settings h3 {
    margin: 2px 0 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--ds-text);
}
input[type="range"] { accent-color: var(--ds-accent); }

/* Documentation tabs */
.ds-doc { line-height: 1.65; color: var(--ds-text); }
.ds-doc h3 {
    margin: 24px 0 10px !important;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--ds-border);
    font-size: 1.12rem !important;
    font-weight: 600 !important;
    color: var(--ds-text);
}
.ds-doc h3:first-child { margin-top: 4px !important; }
.ds-doc ul { padding-left: 1.2rem; }
.ds-doc li { margin: 6px 0; }
.ds-doc hr { border: none; border-top: 1px solid var(--ds-border); margin: 22px 0; }
.ds-doc a { color: var(--ds-accent); font-weight: 500; text-decoration: none; }
.ds-doc a:hover { text-decoration: underline; }
.ds-doc table {
    width: 100%;
    margin: 12px 0 20px;
    border-collapse: separate !important;
    border-spacing: 0;
    border: 1px solid var(--ds-border) !important;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.93rem;
}
.ds-doc th {
    padding: 11px 14px !important;
    background: var(--ds-surface-2) !important;
    border: none !important;
    border-bottom: 1px solid var(--ds-border) !important;
    color: var(--ds-text);
    font-weight: 600;
    text-align: left;
}
.ds-doc td {
    padding: 11px 14px !important;
    border: none !important;
    border-bottom: 1px solid var(--ds-border) !important;
    color: var(--ds-text);
    vertical-align: top;
}
.ds-doc tr:last-child td { border-bottom: none !important; }

/* Footer */
.ds-footer-wrap {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.ds-footer {
    margin-top: 8px;
    padding: 16px 0 4px;
    border-top: 1px solid var(--ds-border);
    color: var(--ds-muted);
    font-size: 0.88rem;
    text-align: center;
}
.ds-footer a { color: var(--ds-accent); font-weight: 500; text-decoration: none; }
.ds-footer a:hover { text-decoration: underline; }

/* Keyboard focus */
a:focus-visible,
button:focus-visible {
    outline: 2px solid var(--ds-accent);
    outline-offset: 2px;
}

/* Small screens */
@media (max-width: 820px) {
    .ds-hero { grid-template-columns: 1fr; padding: 26px 22px; gap: 22px; }
    .ds-hero h1 { font-size: 1.9rem !important; }
    .gradio-container { padding: 16px 12px 8px !important; }
}
"""

HERO_HTML = """
<div class="ds-hero">
<div>
<h1>DumGPT Studio</h1>
<p class="ds-lede">Interactive multi-model playground for the DumGPT family: Base, RLVR, and Distilled models.</p>
<div class="ds-hero-links">
<a class="ds-link ds-link-solid" href="https://huggingface.co/spaces/shubhamisok/dumgpt-studio" target="_blank" rel="noopener">Hugging Face Space</a>
<a class="ds-link" href="https://github.com/ShubhamKumarSahu-svg/DumGPT2_124M" target="_blank" rel="noopener">Source code on GitHub</a>
</div>
</div>
<div class="ds-spec">
<p class="ds-spec-title">DumGPT-124M base model</p>
<dl>
<div class="ds-row"><dt>Architecture</dt><dd>GPT-2 decoder-only</dd></div>
<div class="ds-row"><dt>Parameters</dt><dd>124M</dd></div>
<div class="ds-row"><dt>Layers / heads</dt><dd>12 / 12</dd></div>
<div class="ds-row"><dt>Context window</dt><dd>1,024 tokens</dd></div>
<div class="ds-row"><dt>Training data</dt><dd>FineWeb-Edu</dd></div>
</dl>
</div>
</div>
"""

FOOTER_HTML = """
<div class="ds-footer">
DumGPT Studio is a playground for small language models trained from scratch.
Model weights on <a href="https://huggingface.co/shubhamisok/DumGPT-124M" target="_blank" rel="noopener">Hugging Face</a>,
code on <a href="https://github.com/ShubhamKumarSahu-svg/DumGPT2_124M" target="_blank" rel="noopener">GitHub</a>.
</div>
"""

with gr.Blocks(title="DumGPT Studio", css=custom_css, theme=custom_theme) as demo:
    gr.HTML(HERO_HTML, elem_classes=["ds-hero-wrap"])

    with gr.Tabs():
        with gr.TabItem("Playground"):
            with gr.Row():
                with gr.Column(scale=3):
                    model_selector = gr.Dropdown(
                        choices=list(AVAILABLE_MODELS.keys()),
                        value=DEFAULT_MODEL_NAME,
                        label="Model variant",
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
                        label="Generated output (streaming)",
                        lines=8,
                        elem_classes=["output-box"],
                    )

                with gr.Column(scale=1, elem_classes=["ds-settings"]):
                    gr.Markdown("### Generation settings")
                    with gr.Group():
                        max_tokens_slider = gr.Slider(
                            minimum=10, maximum=256, value=60, step=10, label="Max new tokens"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.8, value=0.8, step=0.05, label="Temperature"
                        )
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=100, value=50, step=5, label="Top-K sampling"
                        )
                        seed_input = gr.Number(
                            value=42, label="Random seed", precision=0
                        )

            gr.Examples(
                examples=[
                    [DEFAULT_MODEL_NAME, "The history of artificial intelligence begins with", 60, 0.8, 50, 42],
                    [DEFAULT_MODEL_NAME, "Once upon a time, in a land far away,", 60, 0.8, 50, 1337],
                    [DEFAULT_MODEL_NAME, "The fundamental law of nature states that", 70, 0.75, 40, 2024],
                    [DEFAULT_MODEL_NAME, "In modern computer science, algorithms are", 60, 0.8, 50, 777],
                ],
                inputs=[model_selector, prompt_input, max_tokens_slider, temperature_slider, top_k_slider, seed_input],
                label="Sample prompts",
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

        with gr.TabItem("Architecture & Roadmap"):
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
                """,
                elem_classes=["ds-doc"],
            )

        with gr.TabItem("Training Curves & Metrics"):
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
                """,
                elem_classes=["ds-doc"],
            )

    gr.HTML(FOOTER_HTML, elem_classes=["ds-footer-wrap"])

if __name__ == "__main__":
    demo.queue().launch()