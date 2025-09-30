"""
Training Codes of LightningDiT together with VA-VAE.
It envolves advanced training methods, sampling methods, 
architecture design methods, computation methods. We achieve
state-of-the-art FID 1.35 on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL
"""

import torch
def _passthrough_torch_compile(fn=None, *args, **kwargs):
    """Disable torch.compile in environments lacking CUDA linking for Inductor."""
    if callable(fn):
        return fn
    def decorator(func):
        return func
    return decorator
if getattr(torch, "compile", None) is not None:
    torch.compile = _passthrough_torch_compile  # type: ignore
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import math
import yaml
import json
import numpy as np
import logging
import os
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import torchvision

from tokenizer.autoencoder import AutoencoderKL as TokenizerKL
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from accelerate import Accelerator
from datasets.img_latent_dataset import PairedImgLatentDataset

def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
    """
    # Setup accelerator:
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tensorboard_dir_log = f"tensorboard_logs/{exp_name}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir_log)

        # add configs to tensorboard
        config_str=json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)
    else:
        writer = None
        logger = create_logger(None)

    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"
    
    # get rank
    rank = accelerator.local_process_index

    # Create model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
    )

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # load pretrained model
    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = load_weights_with_shape_check(model, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")
    requires_grad(ema, False)
    
    # Load VAE
    if accelerator.is_main_process:
        logger.info("Loading VAE ...")
    vae = TokenizerKL(embed_dim=32, ch_mult=(1, 1, 2, 2, 4))
    vae_ckpt = torch.load('weights/vavae-imagenet256-f16d32-dinov2.pt', map_location='cpu')
    if 'state_dict' in vae_ckpt:
        vae_state_dict = vae_ckpt['state_dict']
    else:
        vae_state_dict = vae_ckpt

    filtered_state_dict = {}
    for k, v in vae_state_dict.items():
        if k.startswith("encoder.") or k.startswith("decoder.") or k.startswith("quant_conv.") or k.startswith("post_quant_conv."):
            filtered_state_dict[k] = v

    vae.load_state_dict(filtered_state_dict)
    vae = vae.to(device)
    vae.eval()


    if dist.is_available() and dist.is_initialized():
        model = DDP(model.to(device), device_ids=[rank])
    else:
        model = model.to(device)
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity; 
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
    opt = torch.optim.AdamW(model.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    
    # Setup data
    dataset = PairedImgLatentDataset(
        source_data_dir=train_config['data']['source_data_path'],
        target_data_dir=train_config['data']['target_data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images. Source: {train_config['data']['source_data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
    
    # Get a fixed batch for visualization
    if accelerator.is_main_process:
        logger.info("Getting fixed batch for visualization...")
        fixed_batch = next(iter(loader))
        if accelerator.mixed_precision == 'no':
            fixed_x = fixed_batch['image'].to(device, dtype=torch.float32)
            fixed_cond = fixed_batch['cond'].to(device, dtype=torch.float32)
            fixed_y = fixed_batch['class_label'].to(device)
        else:
            fixed_x = fixed_batch['image'].to(device)
            fixed_cond = fixed_batch['cond'].to(device)
            fixed_y = fixed_batch['class_label'].to(device)

    if accelerator.is_main_process and 'weight_init' in train_config['train'] and train_config['train']['weight_init']:
        logger.info("weight_init is set, calling visualization function to check for errors before starting training.")
        
        # Get train_steps from checkpoint name
        try:
            train_steps_from_ckpt = int(train_config['train']['weight_init'].split('/')[-1].split('.')[0])
        except (ValueError, IndexError):
            logger.warning("Could not parse step number from weight_init path, using 0 for visualization.")
            train_steps_from_ckpt = 0

        visualize_and_save_samples(
            ema, transport, vae, fixed_x, fixed_cond, fixed_y,
            writer, train_steps_from_ckpt, experiment_dir, accelerator, train_config
        )
        logger.info("Visualization check passed.")


    if 'valid_source_path' in train_config['data'] and 'valid_target_path' in train_config['data']:
        valid_dataset = PairedImgLatentDataset(
            source_data_dir=train_config['data']['valid_source_path'],
            target_data_dir=train_config['data']['valid_target_path'],
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images.")

    # Prepare models for training:
    update_ema(ema, model.module if hasattr(model, "module") else model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    train_config['train']['resume'] = train_config['train']['resume'] if 'resume' in train_config['train'] else False

    if train_config['train']['resume']:
        # check if the checkpoint exists
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getsize(x))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            # opt.load_state_dict(checkpoint['opt'])
            ema.load_state_dict(checkpoint['ema'])
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")

    while True:
        for batch in loader:
            if accelerator.mixed_precision == 'no':
                x = batch['image'].to(device, dtype=torch.float32)
                cond = batch['cond'].to(device, dtype=torch.float32)
                y = batch['class_label'].to(device)
            else:
                x = batch['image'].to(device)
                cond = batch['cond'].to(device)
                y = batch['class_label'].to(device)

            model_kwargs = dict(y=y, cond=cond)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            if 'cos_loss' in loss_dict:
                mse_loss = loss_dict["loss"].mean()
                loss = loss_dict["cos_loss"].mean() + mse_loss
            else:
                loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            update_ema(ema, model.module if hasattr(model, "module") else model)

            # Log loss values:
            if 'cos_loss' in loss_dict:
                running_loss += mse_loss.item()
            else:
                running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes if distributed is initialized
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    world_size = dist.get_world_size()
                    avg_loss = avg_loss.item() / max(world_size, 1)
                else:
                    avg_loss = avg_loss.item()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    logger.info(f"Saving checkpoint...")
                    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    checkpoint = {
                        "model": model_state,
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    # Visualize and save samples
                    logger.info("Generating and saving samples...")
                    visualize_and_save_samples(
                        ema, transport, vae, fixed_x, fixed_cond, fixed_y,
                        writer, train_steps, experiment_dir, accelerator, train_config
                    )

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

                # Evaluate on validation set
                if 'valid_path' in train_config['data']:
                    if accelerator.is_main_process:
                        logger.info(f"Start evaluating at step {train_steps}")
                    val_loss = evaluate(model, valid_loader, device, transport, (0.0, 1.0))
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                        world_size = dist.get_world_size()
                        val_loss = val_loss.item() / max(world_size, 1)
                    else:
                        val_loss = val_loss.item()
                    if accelerator.is_main_process:
                        logger.info(f"Validation Loss: {val_loss:.4f}")
                        writer.add_scalar('Loss/validation', val_loss, train_steps)
                    model.train()
            if train_steps >= train_config['train']['max_steps']:
                break
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator

@torch.no_grad()
def visualize_and_save_samples(
    model, transport, vae, x, cond, y, writer,
    current_step, output_dir, accelerator, train_config
):
    """
    Generates and saves a grid of image samples from the model.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # De-normalize latents for VAE decoding
    latent_multiplier = train_config['data'].get('latent_multiplier', 0.18215)
    latent_norm = train_config['data'].get('latent_norm', False)

    x_for_vae = x / latent_multiplier
    cond_for_vae = cond / latent_multiplier
    
    if latent_norm:
        source_stats_path = os.path.join(train_config['data']['source_data_path'], 'latents_stats.pt')
        target_stats_path = os.path.join(train_config['data']['target_data_path'], 'latents_stats.pt')
        if not os.path.exists(source_stats_path) or not os.path.exists(target_stats_path):
            pass
        else:
            source_stats = torch.load(source_stats_path, map_location='cpu')
            target_stats = torch.load(target_stats_path, map_location='cpu')
            source_mean, source_std = source_stats['mean'].to(x.device), source_stats['std'].to(x.device)
            target_mean, target_std = target_stats['mean'].to(x.device), target_stats['std'].to(x.device)

            cond_for_vae = cond_for_vae * source_std + source_mean
            x_for_vae = x_for_vae * target_std + target_mean

    # Generate samples
    model_kwargs = dict(y=y, cond=cond) # model still needs normalized latents
    sampler = Sampler(transport)
    z = torch.randn_like(x)
    
    ode_sampler = sampler.sample_ode()
    sample_fn = ode_sampler(z, model, **model_kwargs)
    
    # The final sample is at the last step of the ODE solve
    sample = sample_fn[-1] # this sample is normalized

    sample_for_vae = sample / latent_multiplier
    if latent_norm and os.path.exists(target_stats_path): # check again
        # The model output `sample` is in the same normalized space as `x`
        sample_for_vae = sample_for_vae * target_std + target_mean

    # Decode latents to images
    x_image = vae.decode(x_for_vae).clamp(-1, 1)
    cond_image = vae.decode(cond_for_vae).clamp(-1, 1)
    sample_image = vae.decode(sample_for_vae).clamp(-1, 1)

    # Create a grid of images
    # We will show Condition, Prediction, and Ground Truth
    grid = torchvision.utils.make_grid(
        torch.cat([cond_image, sample_image, x_image], dim=0),
        nrow=x.size(0),
        normalize=True,
    )

    # Save the image grid
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    torchvision.utils.save_image(grid, os.path.join(sample_dir, f"step_{current_step:07d}.png"))

    # Log to TensorBoard
    writer.add_image("Samples (Condition-Prediction-GroundTruth)", grid, global_step=current_step)

    # Set the model back to training mode
    model.train()


def load_weights_with_shape_check(model, checkpoint, rank=0):
    
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # special case for x_embedder.proj.weight
                # the pretrained model is trained with 256x256 images
                # we can load the weights by resizing the weights
                # and keep the first 3 channels the same
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
                model_state_dict[name] = weight
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    Safe to call before torch.distributed init; defaults to rank 0 behavior.
    """
    is_main = True
    if dist.is_available() and dist.is_initialized():
        try:
            is_main = dist.get_rank() == 0
        except Exception:
            is_main = True
    if is_main and logging_dir is not None:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lightningdit_xl_vavae_f16d32.yaml')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_train(train_config, accelerator)
