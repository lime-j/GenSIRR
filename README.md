# GenSIRR

Official code release for the CVPR 2026 paper:

**Rectifying Latent Space for Generative Single-Image Reflection Removal**  
Mingjia Li, Jin Hu, Hainuo Wang, Qiming Hu, Jiarui Wang, Xiaojie Guo

- Paper: https://arxiv.org/html/2512.06358v1
- Project page: https://gensirr.research.mingjia.li
- Checkpoint: https://huggingface.co/lime-j/GenSIRR

GenSIRR adapts a pretrained image-editing latent diffusion model to single-image
reflection removal. The release includes the Gradio demo, FLUX/DiT training code,
VAE training code, and user-study analysis utilities.

## Method Overview

GenSIRR is built around three components:

- Reflection-equivariant VAE for aligning the latent space with the linear
  superposition model of reflection formation.
- Learnable task-specific text embedding for reflection removal guidance without
  relying on ambiguous natural-language prompts.
- Depth-guided early-branching sampling for selecting stronger candidates on
  difficult real-world inputs.

## Repository Layout

```text
demo/             Gradio app, GenSIRR inference pipeline, demo embeddings
flux_training/    FLUX/Kontext training and inference code for GenSIRR
vae_training/     Reflection-equivariant VAE training scripts
user_study/       User-study labeling and aggregation utilities
README.md         Release documentation
```

## Demo

The demo runs GenSIRR on top of `black-forest-labs/FLUX.1-Kontext-dev` and
downloads `GenSIRR.pt` from the Hugging Face repo at runtime.

Install dependencies in a CUDA environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r demo/requirements.txt
pip install gradio spaces huggingface_hub pillow numpy
```

Run:

```bash
python demo/app.py
```

The app loads:

- `demo/prompt_embeds.pth`
- `demo/pooled_prompt_embeds.pth`
- `demo/text_ids.pth`
- `lime-j/GenSIRR/GenSIRR.pt`

The main `GenSIRR.pt` checkpoint used by the demo contains the model state
needed for demo inference. The standalone trained GenSIRR VAE is released at
https://huggingface.co/lime-j/GenSIRR-vae for training or custom inference
workflows that need the VAE separately.

## Data Preparation

Most SIRR training and evaluation data can be downloaded from:

```bash
wget https://checkpoints.mingjia.li/sirs.zip
unzip sirs.zip
```

This archive contains the main SIRR training data and the Nature, Real, and SIR2
evaluation data used by the training config. The expected layout is:

```text
sirs/
  train/
    real/
    nature/
    VOCdevkit/VOC2012/PNGImages/
    VOC2012_224_train_png.txt
    rrw/
  test/
    real20_420/
    Nature/
    SIR2/
      SolidObjectDataset/
      PostcardDataset/
      WildSceneDataset/
```

RRW is distributed separately. Download it from either:

- https://checkpoints.mingjia.li/rrw.7z
- https://huggingface.co/datasets/zhuyr/RRW

Then place the extracted RRW folder next to `sirs/` and run the preprocessing
script from `flux_training/`:

```bash
cd flux_training
# Expected relative paths:
#   ../rrw  -> extracted RRW source folder
#   ../sirs -> SIRR data root from sirs.zip
python preprocess_rrw.py
```


The script copies paired RRW images into:

```text
sirs/train/rrw/
  blended/
  transmission_layer/
```

Thanks [@krantbrity](https://github.com/krantbrity) for this awesome script!

## FLUX Training and Inference

The FLUX training code lives in `flux_training/` and uses Lightning with
DeepSpeed. A public template config is provided at:

```text
flux_training/options/gensirr.yml
```

Before training, edit the placeholder paths in that file:

- `network_g.model_path`: local path or Hugging Face id for FLUX.1 Kontext
- `network_g.vae_path`: local path to the trained GenSIRR VAE, or `lime-j/GenSIRR-vae`
- dataset `datadir` and `fns` fields
- `path.experiments_root` if you want outputs outside `flux_training/experiments`

Example:

```bash
cd flux_training
pip install -e .
python3 xreflection/tools/train.py --config options/gensirr.yml
```

For inference with a trained DeepSpeed checkpoint:

```bash
cd flux_training
python3 xreflection/tools/infer_flux.py \
  --config options/gensirr.yml \
  --checkpoint /path/to/last.ckpt \
  --input /path/to/input_images \
  --output inference_results \
  --steps 28 \
  --short-side 768
```

Use `infer_flux.py` for final quantitative/qualitative results. The validation
results produced inside the `xreflection` Lightning training loop are intended
only to track training trends, because training-time validation avoids the
slower full-resolution inference path.

For final evaluation, run full inference with `--short-side 512` and/or
`--short-side 768`, then choose the better setting per dataset. We use the best
of the two short-side resolutions for each dataset when reporting final results.

By default, the training and inference code loads the released prompt embedding
files from `demo/`. To use another location, set:

```bash
export GENSIRR_EMBEDDING_DIR=/path/to/embedding_files
```

## VAE Training

`vae_training/` contains the VAE training code used for the
reflection-equivariant latent-space rectification stage. The original training is conducted with Google TPU v4-64, thanks to Google TRC's generous donation. We use [PD-12M](https://huggingface.co/datasets/Spawning/pd12m-full), a high-quality image dataset, as the training set. 

Main files:

- `vae_training/train.py`: CUDA training script
- `vae_training/train_tpu.py`: TPU/XLA training script

The original VAE initialization is the FLUX.1 Kontext VAE:

```text
black-forest-labs/FLUX.1-Kontext-dev/vae/diffusion_pytorch_model.safetensors
```

For the current VAE training scripts, download or copy that file to:

```text
vae_training/flux_vae.safetensors
```

The trained GenSIRR VAE is released separately at
https://huggingface.co/lime-j/GenSIRR-vae. The current VAE data loader follows
the original WebDataset/`gsutil` workflow used during development, so dataset
access must be configured locally before launching VAE training.

## User Study

The user-study utilities are in `user_study/`.

```bash
python user_study/analyze_user_study.py
```

This aggregates the raw label files under `user_study/raw_user_study_results/`
and reports per-method success and failure rates.

## Acknowledgement 

The author would like to express their gratitude to Google TPU Research Cloud (TRC) for their generous donation of computational resources.

## Citation

If you use this code or checkpoint, please cite:

```bibtex
@InProceedings{Li_2026_CVPR,
    author    = {Li, Mingjia and Hu, Jin and Wang, Hainuo and Hu, Qiming and Wang, Jiarui and Guo, Xiaojie},
    title     = {Rectifying Latent Space for Generative Single-Image Reflection Removal},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {8397-8407}
}
```
