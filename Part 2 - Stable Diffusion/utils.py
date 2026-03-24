"""
Utility functions for Part 2 — Stable Diffusion LoRA training.
"""

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file
from tqdm.auto import tqdm


def load_config(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)


def save_lora_checkpoint(unet, save_dir):
    """
    Save LoRA adapter weights in diffusers format (pytorch_lora_weights.safetensors).

    unet.save_pretrained() saves the full UNet (~5 GB); this extracts only the
    LoRA matrices and converts key names so the file can be loaded with
    StableDiffusionXLPipeline.load_lora_weights().
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lora_state = {}
    for k, v in unet.state_dict().items():
        if "lora_A" in k:
            lora_state["unet." + k.replace(".lora_A.default.", ".lora.down.")] = v
        elif "lora_B" in k:
            lora_state["unet." + k.replace(".lora_B.default.", ".lora.up.")] = v

    out = save_dir / "pytorch_lora_weights.safetensors"
    save_file(lora_state, str(out))
    return out


@torch.no_grad()
def encode_prompt(captions, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, device):
    """Encode captions with both SDXL CLIP encoders. Returns (prompt_embeds, pooled_embeds)."""
    def _tok(tokenizer, ids):
        return tokenizer(
            ids, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)

    out1 = text_encoder_1(_tok(tokenizer_1, captions), output_hidden_states=True)
    out2 = text_encoder_2(_tok(tokenizer_2, captions), output_hidden_states=True)

    prompt_embeds = torch.cat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1)
    return prompt_embeds, out2[0]


def get_add_time_ids(batch_size, device, dtype, image_size=1024):
    """Build SDXL's additional conditioning tensor (orig size, crop offset, target size)."""
    ids = [image_size, image_size, 0, 0, image_size, image_size]
    return torch.tensor([ids] * batch_size, dtype=dtype, device=device)


def train_epoch(unet, dataloader, optimizer, noise_scheduler, vae,
                tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2,
                device, dtype, epoch_idx, style_name, image_size=1024):
    """Run one epoch of LoRA fine-tuning. Returns average MSE loss."""
    unet.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch_idx} [{style_name}]", leave=False):
        images   = batch["image"].to(device, dtype=dtype)
        captions = batch["caption"]
        bsz      = images.shape[0]

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

        noise     = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        prompt_embeds, pooled_embeds = encode_prompt(
            captions, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, device
        )
        add_time_ids = get_add_time_ids(bsz, device, dtype, image_size=image_size)

        noise_pred = unet(
            noisy_latents, timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def plot_loss_curve(loss_history, style_name, color="steelblue"):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", color=color)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(f"{style_name.replace('_', ' ')} — training loss")
    plt.tight_layout(); plt.show()


def plot_style_sweep(pipe, device, prompt,
                     negative_prompt="blurry, low quality, watermark, text",
                     n_steps=30, guidance_scale=7.5, seed=42,
                     sweep_weights=None, save_path="style_sweep.png", image_size=1024):
    """Generate a grid sweeping from Impressionism to Ukiyo-e."""
    if sweep_weights is None:
        sweep_weights = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]

    images = []
    for w_imp, w_ukiyo in sweep_weights:
        pipe.set_adapters(["impressionism", "ukiyo_e"], adapter_weights=[w_imp, w_ukiyo])
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            img = pipe(prompt=prompt, negative_prompt=negative_prompt,
                       num_inference_steps=n_steps, guidance_scale=guidance_scale,
                       height=image_size, width=image_size, generator=generator).images[0]
        images.append((w_imp, w_ukiyo, img))
        print(f"  [{w_imp:.2f} Impr. | {w_ukiyo:.2f} Ukiyo] ✓")

    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 5, 5.5))
    for ax, (w_imp, w_ukiyo, img) in zip(axes, images):
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"Impr {w_imp:.2f}\nUkiyo {w_ukiyo:.2f}", fontsize=10)
    fig.suptitle(f'Style sweep — "{prompt}"', fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def generate_styled(pipe, device, prompt, impressionism_weight=0.5,
                    n_images=4, n_steps=30, guidance_scale=7.5, seed=42,
                    negative_prompt="blurry, low quality, watermark, text", image_size=1024):
    """Generate images by blending Impressionism and Ukiyo-e LoRA adapters."""
    ukiyo_weight = 1.0 - impressionism_weight
    pipe.set_adapters(["impressionism", "ukiyo_e"], adapter_weights=[impressionism_weight, ukiyo_weight])

    images = []
    for i in range(n_images):
        generator = torch.Generator(device=device).manual_seed(seed + i)
        with torch.no_grad():
            img = pipe(prompt=prompt, negative_prompt=negative_prompt,
                       num_inference_steps=n_steps, guidance_scale=guidance_scale,
                       height=image_size, width=image_size, generator=generator).images[0]
        images.append(img)

    ncols = min(n_images, 4)
    nrows = math.ceil(n_images / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes_flat = np.array(axes).reshape(-1) if n_images > 1 else [axes]
    for ax, img in zip(axes_flat, images):
        ax.imshow(img); ax.axis("off")
    for ax in axes_flat[len(images):]:
        ax.axis("off")

    if impressionism_weight == 1.0:   label = "Pure Impressionism"
    elif impressionism_weight == 0.0: label = "Pure Ukiyo-e"
    else: label = f"Blend [{impressionism_weight:.2f} Impr. | {ukiyo_weight:.2f} Ukiyo]"
    fig.suptitle(f'{label}\n"{prompt}"', fontsize=11)
    plt.tight_layout(); plt.show()
