# Diffusion Models — Hands-On Tutorial

A two-part tutorial built for **IndabaX Uganda 2026**, taking you from implementing a DDPM from scratch to fine-tuning Stable Diffusion XL with LoRA.

---

## Part 1 — DDPM from Scratch

Build a Denoising Diffusion Probabilistic Model (DDPM) and train it on CelebA-HQ with gender conditioning.

**What you'll build:**

- A U-Net architecture (`DDPMUnet`) with residual blocks and embedding layers
- Forward (noising) and reverse (denoising) diffusion processes
- Context-conditioned generation via a gender embedding
- An interactive slider to blend between female/male conditioning at inference

**Pre-trained weights:**

- [Unconditional DDPM](https://drive.google.com/file/d/1npadlxlYPQiHbkU4cGapPa6rXlwFiYTU/view?usp=sharing)
- [Context-conditioned DDPM](https://drive.google.com/file/d/1pjQyghEyO9REO_3HEH0HajC2_YES2L1l/view?usp=sharing)

**Dataset:** [CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) — 1024×1024 face images

---

## Part 2 — Stable Diffusion & LoRA

Fine-tune SDXL using Low-Rank Adaptation (LoRA) to learn artistic styles, then blend them at inference.

**What you'll build:**

- LoRA adapters for two distinct art styles (Impressionism, Ukiyo-e) trained on WikiArt
- Style mixing at inference by interpolating adapter weights — the direct analogue of the context slider from Part 1
- Combined text-prompt + style-weight control

**Pre-trained LoRA weights:**

- [Impressionism LoRA](https://drive.google.com/file/d/1_4Z_o4dRN1gVqq8fnq2DnIDI7czVLcHc/view?usp=sharing)
- [Ukiyo-e LoRA](https://drive.google.com/file/d/1mKVsPNkq-WRCjXHvXI9ck2G_eSn6a_Ao/view?usp=sharing)

**Base model:** [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) — 1024×1024
**Dataset:** [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart)

---

## Setup

```bash
pip install -r requirements.txt
```

Then open the notebook for each part:

```
Part 1 - DDPM/tutorial.ipynb
Part 2 - Stable Diffusion/tutorial.ipynb
```
