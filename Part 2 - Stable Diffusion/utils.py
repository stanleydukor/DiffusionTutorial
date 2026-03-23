"""
Utility functions

This module provides helper functions for:
- Loading configuration
- Encoding text prompts for SDXL
- Training a single LoRA epoch
- Visualising loss curves and generated style sweeps
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm


# ── Configuration ─────────────────────────────────────────────────────────────

def load_config(config_file):
    """
    Load configuration from YAML file

    Args:
        config_file: path to YAML config file
    Returns:
        dict: configuration parameters
    """
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


# ── SDXL text & time conditioning ─────────────────────────────────────────────

@torch.no_grad()
def encode_prompt(captions, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, device):
    """
    Encode text captions with both SDXL CLIP text encoders

    Args:
        captions: list of caption strings
        tokenizer_1: CLIPTokenizer for the first encoder
        tokenizer_2: CLIPTokenizer for the second encoder
        text_encoder_1: CLIPTextModel (ViT-L)
        text_encoder_2: CLIPTextModelWithProjection (ViT-bigG)
        device: "cuda" or "cpu"
    Returns:
        prompt_embeds: concatenated hidden states [B, 77, 2048]
        pooled_embeds: pooled output from second encoder [B, 1280]
    """
    tok1 = tokenizer_1(
        captions,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    tok2 = tokenizer_2(
        captions,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    out1 = text_encoder_1(tok1, output_hidden_states=True)
    out2 = text_encoder_2(tok2, output_hidden_states=True)

    # Second-to-last hidden states (SDXL convention)
    hidden1 = out1.hidden_states[-2]          # [B, 77, 768]
    hidden2 = out2.hidden_states[-2]          # [B, 77, 1280]
    prompt_embeds = torch.cat([hidden1, hidden2], dim=-1)   # [B, 77, 2048]

    # Pooled output from the second encoder
    pooled_embeds = out2[0]                   # [B, 1280]

    return prompt_embeds, pooled_embeds


def get_add_time_ids(batch_size, device, dtype, image_size=1024):
    """
    Build SDXL's additional conditioning tensor (original_size, crop_offset, target_size)

    Args:
        batch_size: number of images in the batch
        device: "cuda" or "cpu"
        dtype: model weight dtype
        image_size: target image resolution (default 1024)
    Returns:
        tensor of shape [B, 6]
    """
    ids = [image_size, image_size, 0, 0, image_size, image_size]
    return torch.tensor([ids] * batch_size, dtype=dtype, device=device)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(
    unet,
    dataloader,
    optimizer,
    noise_scheduler,
    vae,
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    device,
    dtype,
    epoch_idx,
    style_name,
    image_size=1024,
):
    """
    Run one epoch of LoRA fine-tuning on a single artistic style

    Args:
        unet: UNet with LoRA adapters injected
        dataloader: DataLoader yielding {"image": Tensor, "caption": str}
        optimizer: AdamW or similar
        noise_scheduler: DDPMScheduler
        vae: AutoencoderKL (frozen)
        tokenizer_1, tokenizer_2: CLIPTokenizers
        text_encoder_1, text_encoder_2: CLIPTextModels (frozen)
        device: "cuda" or "cpu"
        dtype: model weight dtype
        epoch_idx: current epoch number (for progress display)
        style_name: style name (for progress display)
    Returns:
        average MSE loss over the epoch
    """
    unet.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch_idx} [{style_name}]", leave=False):
        images   = batch["image"].to(device, dtype=dtype)
        captions = batch["caption"]
        bsz      = images.shape[0]

        # 1. Encode images → latents (VAE spatial compression: 1024 → 128)
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor   # unit-variance scaling

        # 2. Sample noise and random timesteps
        noise     = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
        ).long()

        # 3. Forward diffusion: add noise at the sampled timestep
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 4. Encode text captions with both CLIP encoders
        prompt_embeds, pooled_embeds = encode_prompt(
            captions, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, device
        )

        # 5. SDXL additional conditioning (image size metadata)
        add_time_ids      = get_add_time_ids(bsz, device, dtype, image_size=image_size)
        added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

        # 6. Predict noise with LoRA-augmented UNet
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # 7. MSE loss against actual noise (epsilon prediction)
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # 8. Backprop — only LoRA matrices have requires_grad=True
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in unet.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_loss_curve(loss_history, style_name, color="steelblue", save_dir=None):
    """
    Plot training loss history for a single LoRA style

    Args:
        loss_history: list of average loss values, one per epoch
        style_name: style name used in the plot title
        color: line colour
        save_dir: if provided, save the plot as a PNG in this directory
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{style_name.replace('_', ' ')} LoRA — training loss")
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"loss_{style_name.lower()}.png")
        plt.savefig(path, dpi=120)
        print(f"Loss curve saved: {path}")

    plt.show()


def plot_style_sweep(
    pipe,
    device,
    prompt,
    negative_prompt="blurry, low quality, watermark, signature, text, distorted",
    n_steps=30,
    guidance_scale=7.5,
    seed=42,
    sweep_weights=None,
    save_path="style_sweep.png",
    image_size=1024,
):
    """
    Generate and display a grid of images sweeping from one LoRA style to another

    Args:
        pipe: StableDiffusionXLPipeline with both LoRA adapters loaded
        device: "cuda" or "cpu"
        prompt: content description, kept the same across all images
        negative_prompt: what to avoid in the output
        n_steps: number of DDIM inference steps
        guidance_scale: classifier-free guidance strength
        seed: random seed, same for all images to isolate the style effect
        sweep_weights: list of (impressionism_weight, ukiyo_weight) tuples.
                       Defaults to a 5-step sweep from [1.0, 0.0] to [0.0, 1.0]
        save_path: path to save the output PNG (None = do not save)
    """
    if sweep_weights is None:
        sweep_weights = [
            (1.00, 0.00),
            (0.75, 0.25),
            (0.50, 0.50),
            (0.25, 0.75),
            (0.00, 1.00),
        ]

    print(f"Generating style sweep ({len(sweep_weights)} images) …")
    print(f"Prompt: \"{prompt}\"\n")

    images = []
    for w_imp, w_ukiyo in sweep_weights:
        pipe.set_adapters(
            ["impressionism", "ukiyo_e"],
            adapter_weights=[w_imp, w_ukiyo],
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            img = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=n_steps,
                guidance_scale=guidance_scale,
                height=image_size, width=image_size,
                generator=generator,
            ).images[0]
        images.append((w_imp, w_ukiyo, img))
        print(f"  [{w_imp:.2f} Impr. | {w_ukiyo:.2f} Ukiyo-e] ✓")

    # Plot
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 5, 5.5))
    for ax, (w_imp, w_ukiyo, img) in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Impr {w_imp:.2f}\nUkiyo {w_ukiyo:.2f}", fontsize=10, pad=6)

    fig.suptitle(
        f'Style sweep — same prompt, different LoRA weights\n"{prompt}"',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {save_path}")

    plt.show()


def generate_styled(
    pipe,
    device,
    prompt,
    impressionism_weight=0.5,
    n_images=4,
    n_steps=30,
    guidance_scale=7.5,
    seed=42,
    negative_prompt="blurry, low quality, watermark, text, distorted",
    image_size=1024,
):
    """
    Generate images by blending Impressionism and Ukiyo-e LoRA adapters

    Args:
        pipe: StableDiffusionXLPipeline with both LoRA adapters loaded
        device: "cuda" or "cpu"
        prompt: content description (subject matter)
        impressionism_weight: blend weight in [0, 1]; 1.0 = pure Impressionism,
                              0.0 = pure Ukiyo-e
        n_images: number of images to generate
        n_steps: number of DDIM inference steps
        guidance_scale: classifier-free guidance strength
        seed: base random seed (each image uses seed + i)
        negative_prompt: what to avoid in the output
    """
    ukiyo_e_weight = 1.0 - impressionism_weight
    pipe.set_adapters(
        ["impressionism", "ukiyo_e"],
        adapter_weights=[impressionism_weight, ukiyo_e_weight],
    )

    print(f"Generating {n_images} image(s) …")
    print(f"  Impressionism weight : {impressionism_weight:.2f}")
    print(f"  Ukiyo-e weight       : {ukiyo_e_weight:.2f}")
    print(f"  Prompt               : {prompt}")

    images = []
    for i in range(n_images):
        generator = torch.Generator(device=device).manual_seed(seed + i)
        with torch.no_grad():
            img = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=n_steps,
                guidance_scale=guidance_scale,
                height=image_size, width=image_size,
                generator=generator,
            ).images[0]
        images.append(img)

    # Display grid
    ncols = min(n_images, 4)
    nrows = math.ceil(n_images / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes_flat = np.array(axes).reshape(-1) if n_images > 1 else [axes]

    for ax, img in zip(axes_flat, images):
        ax.imshow(img)
        ax.axis("off")
    for ax in axes_flat[len(images):]:
        ax.axis("off")

    if impressionism_weight == 1.0:
        style_label = "Pure Impressionism"
    elif impressionism_weight == 0.0:
        style_label = "Pure Ukiyo-e"
    else:
        style_label = (
            f"Blend  [{impressionism_weight:.2f} Impr. | {ukiyo_e_weight:.2f} Ukiyo-e]"
        )

    fig.suptitle(f"{style_label}\n\"{prompt}\"", fontsize=11)
    plt.tight_layout()
    plt.show()
