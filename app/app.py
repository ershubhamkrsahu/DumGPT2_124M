import os
import random
import time
import torch
import torch.nn.functional as F
import gradio as gr
from model import load_dumgpt, GPTConfig

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading DumGPT-124M on {device}...")
model, enc, config = load_dumgpt("shubhamisok/DumGPT-124M", device=device)
print("Model ready!")


def generate_stream(
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 42,
):
    if not prompt or not prompt.strip():
        yield "Please provide a prompt to start generation."
        return

    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
        random.seed(seed)

    tokens = enc.encode(prompt)
    if len(tokens) > config.block_size - 10:
        tokens = tokens[-(config.block_size - 10):]

    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated_text = prompt
    start_time = time.time()

    yield generated_text

    with torch.no_grad():
        for i in range(int(max_new_tokens)):
            # Crop to context window
            x_cond = x[:, -config.block_size:]
            logits = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)

            # Optional Top-K filtering
            if top_k > 0:
                topk_probs, topk_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                probs = F.softmax(topk_probs, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                next_token = topk_indices.gather(-1, next_idx)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_token), dim=1)
            new_char = enc.decode([next_token.item()])
            generated_text += new_char

            # Yield periodically for smooth streaming
            if i % 2 == 0 or i == max_new_tokens - 1:
                yield generated_text

    dt = time.time() - start_time
    tok_sec = max_new_tokens / max(dt, 1e-4)
    summary = f"\n\n---\n*Generated {max_new_tokens} tokens in {dt:.2f}s ({tok_sec:.1f} tokens/sec on {device.upper()})*"
    yield generated_text + summary


# Custom theme styling
custom_css = """
.container { max-width: 900px; margin: auto; }
.output-box textarea { font-family: 'Courier New', monospace; font-size: 15px; }
"""

with gr.Blocks(title="DumGPT-124M Studio", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 DumGPT-124M Studio
        **A 124M-parameter GPT-2 language model trained completely from scratch on FineWeb-Edu.**
        
        *Weights & Checkpoints on Hugging Face:* [`shubhamisok/DumGPT-124M`](https://huggingface.co/shubhamisok/DumGPT-124M) • 
        *Source Code:* [`GitHub`](https://github.com/ShubhamKumarSahu-svg/DumGPT2_124M)
        """
    )

    with gr.Tabs():
        with gr.TabItem("🚀 Playground"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Type a beginning of a sentence (e.g. 'The history of artificial intelligence begins with')...",
                        lines=3,
                        value="The history of artificial intelligence begins with",
                    )
                    with gr.Row():
                        generate_btn = gr.Button("Generate", variant="primary", scale=2)
                        clear_btn = gr.Button("Clear", scale=1)

                    output_text = gr.Textbox(
                        label="DumGPT-124M Output (Streaming)",
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
                    ["The history of artificial intelligence begins with", 60, 0.8, 50, 42],
                    ["Once upon a time, in a land far away,", 60, 0.8, 50, 1337],
                    ["The fundamental law of nature states that", 70, 0.75, 40, 2024],
                    ["In modern computer science, algorithms are", 60, 0.8, 50, 777],
                ],
                inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider, seed_input],
                label="Sample Prompts",
            )

            generate_btn.click(
                fn=generate_stream,
                inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider, seed_input],
                outputs=output_text,
            )
            clear_btn.click(lambda: ("", ""), outputs=[prompt_input, output_text])

        with gr.TabItem("📊 Training & Architecture"):
            gr.Markdown(
                r"""
                ### Model Specifications
                - **Architecture**: GPT-2 (Transformer Decoder-only)
                - **Parameters**: 124,475,904 (~124M)
                - **Layers**: 12
                - **Attention Heads**: 12
                - **Embedding Dimension**: 768
                - **Context Window**: 1,024 tokens
                - **Vocabulary Size**: 50,304 (padded to multiple of 128 for tensor core performance)
                - **Weight Tying**: Token embedding `wte` and `lm_head` shared
                
                ### Training Highlights
                - **Pretrained From Scratch**: No OpenAI weights loaded. Initialized from Gaussian noise with loss ~10.8 ($\ln(50,304)$).
                - **Dataset**: `HuggingFaceFW/fineweb-edu` (sample-10BT), tokenized with `tiktoken` (gpt2 encoding) into 100M-token binary uint16 shards.
                - **Optimization**: AdamW with weight decay 0.1 on 2D matrices, warmup to peak LR $6 \times 10^{-4}$ with cosine decay to $6 \times 10^{-5}$, gradient accumulation.
                - **Precision**: Mixed precision (BF16 / FP16) with Flash Attention (SDPA).
                - **Loss**: Training loss dropped to ~6.48 and validation loss to 6.44 at 1,000 steps, transitioning from random noise to coherent grammar and English sentence fragments.
                """
            )

        with gr.TabItem("📈 Metrics & Plots"):
            gr.Markdown(
                """
                ### Training Metrics Plots
                The plots below are generated from `outputs/metrics.csv` during the training run:
                
                | Metric | Plot Link |
                | :--- | :--- |
                | **Training Loss Curve** | [View train_loss.png](https://huggingface.co/shubhamisok/DumGPT-124M/resolve/main/outputs/plots/train_loss.png) |
                | **Validation Loss Curve** | [View val_loss.png](https://huggingface.co/shubhamisok/DumGPT-124M/resolve/main/outputs/plots/val_loss.png) |
                | **Learning Rate Schedule** | [View learning_rate.png](https://huggingface.co/shubhamisok/DumGPT-124M/resolve/main/outputs/plots/learning_rate.png) |
                | **Gradient Norm** | [View grad_norm.png](https://huggingface.co/shubhamisok/DumGPT-124M/resolve/main/outputs/plots/grad_norm.png) |
                | **Tokens / Second** | [View tokens_per_sec.png](https://huggingface.co/shubhamisok/DumGPT-124M/resolve/main/outputs/plots/tokens_per_sec.png) |
                
                *(You can view all raw plots directly in the [Hugging Face Repository](https://huggingface.co/shubhamisok/DumGPT-124M/tree/main/outputs/plots).)*
                """
            )

if __name__ == "__main__":
    demo.queue().launch(share=False)
