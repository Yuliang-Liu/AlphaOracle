"""Single-image repair inference (v3) for LightningDiT conditional models."""

import argparse
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import yaml
from PIL import Image


def _passthrough_torch_compile(fn=None, *args, **kwargs):
    """Disable torch.compile in sandboxed environments."""
    if callable(fn):
        return fn

    def decorator(func):
        return func

    return decorator


if getattr(torch, "compile", None) is not None:
    torch.compile = _passthrough_torch_compile  # type: ignore


from models.lightningdit import LightningDiT_models
from tokenizer.vavae import VA_VAE
from transport import Sampler, create_transport


@dataclass
class InferenceContext:
    device: torch.device
    cfg: dict
    patch_size: int
    downsample_ratio: int
    latent_patch_size: int
    latent_norm: bool
    latent_multiplier: float
    model: torch.nn.Module
    sample_fn: Callable
    vae: VA_VAE
    source_mean: torch.Tensor
    source_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor


@dataclass
class PreparedImage:
    path: str
    orig_width: int
    orig_height: int
    latent: torch.Tensor
    latent_height: int
    latent_width: int


def _auto_device(preferred: str | None = None) -> torch.device:
    if preferred:
        if preferred == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_state_dict(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "ema" in checkpoint:
            state_dict = checkpoint["ema"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _load_latent_stats(root: str) -> tuple[torch.Tensor, torch.Tensor]:
    stats_path = os.path.join(root, "latents_stats.pt")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Latent stats not found at {stats_path}")
    stats = torch.load(stats_path, map_location="cpu")
    return stats["mean"], stats["std"]


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def _prepare_latent(latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, latent_norm: bool, latent_multiplier: float) -> torch.Tensor:
    out = latent
    if latent_norm:
        out = (out - mean) / std
    return out * latent_multiplier


def _recover_latent(latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, latent_norm: bool, latent_multiplier: float) -> torch.Tensor:
    out = latent / latent_multiplier
    if latent_norm:
        out = out * std + mean
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image repair inference (v3) with LightningDiT.")
    parser.add_argument("--config", default="configs/lightningdit_xl_vavae_f16d32.yaml", help="Training config YAML used for the model.")
    parser.add_argument("--checkpoint", default='weights/xxx.pt', help="Checkpoint path for the trained LightningDiT model (EMA preferred).")
    parser.add_argument("--input", default="h20822_11650.jpg", help="Path to the blurry input image.")
    parser.add_argument("--output", default='out.png', help="Where to save the repaired image.")
    parser.add_argument("--vae-config", dest="vae_config", default=None, help="Optional override for tokenizer config path.")
    parser.add_argument("--device", default=None, help="Device to run on (e.g., cuda, cpu). Defaults to auto-detection.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling noise.")
    return parser.parse_args()


def load_inference_context(
    config_path: str,
    checkpoint_path: str,
    *,
    device: Optional[str] = None,
    vae_config_path: Optional[str] = None,
) -> InferenceContext:
    cfg = _load_config(config_path)
    dev = _auto_device(device)

    downsample_ratio = cfg["vae"].get("downsample_ratio", 16)
    patch_size = cfg["data"].get("image_size", 128)
    if patch_size <= 0:
        raise ValueError("Configured image_size must be positive.")
    if patch_size % downsample_ratio != 0:
        raise ValueError(
            f"Configured image_size {patch_size} is incompatible with VAE downsample ratio {downsample_ratio}."
        )

    latent_patch_size = patch_size // downsample_ratio

    model = LightningDiT_models[cfg["model"]["model_type"]](
        input_size=latent_patch_size,
        num_classes=cfg["data"]["num_classes"],
        use_qknorm=cfg["model"].get("use_qknorm", False),
        use_swiglu=cfg["model"].get("use_swiglu", False),
        use_rope=cfg["model"].get("use_rope", False),
        use_rmsnorm=cfg["model"].get("use_rmsnorm", False),
        wo_shift=cfg["model"].get("wo_shift", False),
        in_channels=cfg["model"].get("in_chans", 4),
        use_checkpoint=cfg["model"].get("use_checkpoint", False),
    )
    state_dict = _load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model.to(dev)
    model.eval()

    transport = create_transport(
        cfg["transport"]["path_type"],
        cfg["transport"]["prediction"],
        cfg["transport"].get("loss_weight"),
        cfg["transport"].get("train_eps"),
        cfg["transport"].get("sample_eps"),
        use_cosine_loss=cfg["transport"].get("use_cosine_loss", False),
        use_lognorm=cfg["transport"].get("use_lognorm", False),
    )
    sampler = Sampler(transport)
    if cfg["sample"]["mode"].upper() != "ODE":
        raise NotImplementedError("Single-image v3 script currently supports ODE sampling only.")
    sample_fn = sampler.sample_ode(
        sampling_method=cfg["sample"].get("sampling_method", "dopri5"),
        num_steps=cfg["sample"].get("num_sampling_steps", 50),
        atol=cfg["sample"].get("atol", 1e-6),
        rtol=cfg["sample"].get("rtol", 1e-3),
        reverse=cfg["sample"].get("reverse", False),
        timestep_shift=cfg["sample"].get("timestep_shift", 0.0),
    )

    latent_norm = cfg["data"].get("latent_norm", False)
    latent_multiplier = float(cfg["data"].get("latent_multiplier", 1.0))

    source_mean, source_std = _load_latent_stats(cfg["data"]["source_data_path"])
    target_mean, target_std = _load_latent_stats(cfg["data"]["target_data_path"])
    source_mean = source_mean.to(dev)
    source_std = source_std.to(dev)
    target_mean = target_mean.to(dev)
    target_std = target_std.to(dev)

    vae_cfg = vae_config_path or f"tokenizer/configs/{cfg['vae']['model_name']}.yaml"
    vae = VA_VAE(vae_cfg, dev)

    return InferenceContext(
        device=dev,
        cfg=cfg,
        patch_size=patch_size,
        downsample_ratio=downsample_ratio,
        latent_patch_size=latent_patch_size,
        latent_norm=latent_norm,
        latent_multiplier=latent_multiplier,
        model=model,
        sample_fn=sample_fn,
        vae=vae,
        source_mean=source_mean,
        source_std=source_std,
        target_mean=target_mean,
        target_std=target_std,
    )


def _prepare_image(ctx: InferenceContext, input_path: str) -> PreparedImage:
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        orig_width, orig_height = img.size
        resized_img = img.resize((ctx.patch_size, ctx.patch_size), Image.BICUBIC)

    image_tensor = _image_to_tensor(resized_img).unsqueeze(0).to(ctx.device)
    with torch.no_grad():
        cond_latent = ctx.vae.encode(image_tensor)

    cond_latent = _prepare_latent(
        cond_latent,
        ctx.source_mean,
        ctx.source_std,
        latent_norm=ctx.latent_norm,
        latent_multiplier=ctx.latent_multiplier,
    ).detach()

    latent_height = cond_latent.shape[-2]
    latent_width = cond_latent.shape[-1]

    return PreparedImage(
        path=input_path,
        orig_width=orig_width,
        orig_height=orig_height,
        latent=cond_latent,
        latent_height=latent_height,
        latent_width=latent_width,
    )


def _decode_and_restore(
    ctx: InferenceContext,
    repaired_latent: torch.Tensor,
    prepared: PreparedImage,
    *,
    restore_original_size: bool = True,
) -> Image.Image:
    _ = restore_original_size  # 兼容旧参数，但始终还原为原尺寸
    latent = _recover_latent(
        repaired_latent,
        ctx.target_mean,
        ctx.target_std,
        latent_norm=ctx.latent_norm,
        latent_multiplier=ctx.latent_multiplier,
    )
    decoded_np = ctx.vae.decode_to_images(latent)[0].astype(np.float32)
    decoded_np = np.clip(decoded_np, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(decoded_np)

    if output_image.width != prepared.orig_width or output_image.height != prepared.orig_height:
        output_image = output_image.resize((prepared.orig_width, prepared.orig_height), Image.BICUBIC)

    return output_image


def run_inference_on_batch(
    ctx: InferenceContext,
    input_paths: List[str],
    output_paths: List[str],
    *,
    restore_original_size: bool = True,
) -> None:
    if len(input_paths) != len(output_paths):
        raise ValueError("input_paths and output_paths must have the same length")
    if not input_paths:
        return

    prepared_images = [_prepare_image(ctx, path) for path in input_paths]

    reference = prepared_images[0]
    latent_height = reference.latent_height
    latent_width = reference.latent_width

    for prep in prepared_images[1:]:
        if prep.latent_height != latent_height or prep.latent_width != latent_width:
            raise ValueError("All prepared images in a batch must share the same latent dimensions")

    device = ctx.device
    batch_size = len(prepared_images)
    cond_stack = torch.cat([prep.latent for prep in prepared_images], dim=0).to(device)

    def conditioned_forward(x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, cond=None, **kwargs):
        if y is None:
            y = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return ctx.model(x, t, y, cond=cond_stack)

    noise_full = torch.randn(batch_size, ctx.model.in_channels, latent_height, latent_width, device=device)
    y_label = torch.zeros(batch_size, dtype=torch.long, device=device)

    with torch.no_grad():
        latent_trajectory = ctx.sample_fn(noise_full, conditioned_forward, y=y_label)
        repaired_latent_norm = latent_trajectory[-1]

    for idx, (prep, output_path) in enumerate(zip(prepared_images, output_paths)):
        repaired_latent_sample = repaired_latent_norm[idx : idx + 1]
        output_image = _decode_and_restore(
            ctx,
            repaired_latent_sample,
            prep,
            restore_original_size=restore_original_size,
        )

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_image.save(output_path)
        print(f"Saved repaired image to {output_path}")


def run_inference_on_image(
    ctx: InferenceContext,
    input_path: str,
    output_path: str,
    *,
    restore_original_size: bool = True,
) -> None:
    run_inference_on_batch(
        ctx,
        [input_path],
        [output_path],
        restore_original_size=restore_original_size,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    ctx = load_inference_context(
        args.config,
        args.checkpoint,
        device=args.device,
        vae_config_path=args.vae_config,
    )
    run_inference_on_image(
        ctx,
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()
