"""Model utilities for Part 2 — Stable Diffusion LoRA training."""

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def load_sdxl_components(model_id, dtype, device):
    """Load all SDXL sub-models. Text encoders and VAE are frozen; only UNet trains."""
    print("Loading scheduler …")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    print("Loading tokenizers …")
    tokenizer_1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")

    print("Loading text encoders …")
    text_encoder_1 = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)

    print("Loading VAE …")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)

    print("Loading UNet …")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    ).to(device)

    for model in (text_encoder_1, text_encoder_2, vae):
        model.requires_grad_(False)
        model.eval()

    print("✓ All components loaded — text encoders & VAE frozen.")
    return noise_scheduler, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, vae, unet


def configure_lora(unet, rank=16, alpha=16, target_modules=None):
    """Inject LoRA adapters into UNet attention layers; freeze all base weights."""
    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    unet.add_adapter(lora_config)
    unet.enable_gradient_checkpointing()

    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"LoRA injected  : rank={rank}, alpha={alpha}, scale={alpha / rank:.2f}")
    print(f"Target modules : {target_modules}")
    print(f"Trainable      : {trainable:,} / {total:,}  ({100 * trainable / total:.3f}%)")

    return unet, lora_config
