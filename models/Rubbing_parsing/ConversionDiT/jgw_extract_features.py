import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
from tokenizer.vavae import VA_VAE


def main(args):
    """
    Encode paired (condition, target) images into VA-VAE latents with strict per-sample alignment.
    Random/forced flips are disabled to avoid misalignment.
    """
    assert torch.cuda.is_available(), "Extract features currently requires at least one GPU."

    # ============ DDP ============ #
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # ============ IO ============ #
    cond_path = args.cond_path if args.cond_path else args.data_path
    target_path = args.target_path if args.target_path else args.data_path

    output_dir = os.path.join(
        args.output_path,
        os.path.splitext(os.path.basename(args.config))[0],
        f'{args.data_split}_{args.image_size}'
    )
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Condition root: {cond_path}")
        print(f"Target root:    {target_path}")
        print(f"Output dir:     {output_dir}")

    # ============ Model ============ #
    tokenizer = VA_VAE(args.config)

    # No random flip
    transform_noflip = tokenizer.img_transform(p_hflip=0.0)

    ds_cond = ImageFolder(cond_path, transform=transform_noflip)
    ds_tgt = ImageFolder(target_path, transform=transform_noflip)

    samplers = [
        DistributedSampler(ds_cond, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed),
        DistributedSampler(ds_tgt, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed),
    ]
    loaders = [
        DataLoader(ds_cond, batch_size=args.batch_size, shuffle=False, sampler=samplers[0],
                   num_workers=args.num_workers, pin_memory=True, drop_last=False),
        DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=False, sampler=samplers[1],
                   num_workers=args.num_workers, pin_memory=True, drop_last=False),
    ]

    total_data_in_loop = min(len(ds_cond), len(ds_tgt))
    if rank == 0:
        print(f"Total paired samples: {total_data_in_loop}")

    run_images, saved_files = 0, 0
    latents_cond, latents_target = [], []
    labels_cond, labels_target = [], []

    for batch_idx, (batch_cond, batch_tgt) in enumerate(zip(*loaders)):
        x_c, y_c = batch_cond
        x_t, y_t = batch_tgt

        # 对齐 batch 尺寸
        min_bs = min(x_c.size(0), x_t.size(0))
        x_c, y_c = x_c[:min_bs], y_c[:min_bs]
        x_t, y_t = x_t[:min_bs], y_t[:min_bs]

        run_images += min_bs
        if (run_images % 100 == 0) and rank == 0:
            print(f'{datetime.now()} processed {run_images}/{total_data_in_loop}')

        with torch.no_grad():
            z_c = tokenizer.encode_images(x_c).detach().cpu()
            z_t = tokenizer.encode_images(x_t).detach().cpu()

        if batch_idx == 0 and rank == 0:
            print('latent_cond shape', z_c.shape)
            print('latent_target shape', z_t.shape)

        latents_cond.append(z_c)
        latents_target.append(z_t)
        labels_cond.append(y_c)
        labels_target.append(y_t)

        if len(latents_cond) == 10000 // args.batch_size:
            save_shard(latents_cond, latents_target, labels_cond, labels_target,
                       output_dir, rank, saved_files)
            latents_cond, latents_target, labels_cond, labels_target = [], [], [], []
            saved_files += 1

    # save remainder
    if len(latents_cond) > 0:
        save_shard(latents_cond, latents_target, labels_cond, labels_target,
                   output_dir, rank, saved_files)
        
    dist.barrier()
    if rank == 0:
        from datasets.img_latent_dataset import ImgLatentDataset
        _ = ImgLatentDataset(output_dir, latent_norm=True)
    dist.barrier()
    dist.destroy_process_group()

    try:
        dist.barrier()
        dist.destroy_process_group()
    except:
        pass


def save_shard(latents_cond, latents_target, labels_cond, labels_target,
               output_dir, rank, shard_idx):
    latents_cond = torch.cat(latents_cond, dim=0)
    latents_target = torch.cat(latents_target, dim=0)
    labels_cond = torch.cat(labels_cond, dim=0)
    labels_target = torch.cat(labels_target, dim=0)

    save_dict = {
        'latents_cond': latents_cond,
        'latents_target': latents_target,
        'labels_cond': labels_cond,
        'labels_target': labels_target,
    }
    if rank == 0:
        for k, v in save_dict.items():
            print(k, v.shape)

    save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{shard_idx:03d}.safetensors')
    save_file(
        save_dict,
        save_filename,
        metadata={'total_size': f'{latents_cond.shape[0]}',
                  'dtype': f'{latents_cond.dtype}',
                  'device': f'{latents_cond.device}'}
    )
    if rank == 0:
        print(f'Saved {save_filename}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond_path", type=str, default='path/to/source', help="Root of condition images (ImageFolder).")
    parser.add_argument("--target_path", type=str, default='path/to/target', help="Root of target images (ImageFolder).")
    parser.add_argument("--data_path", type=str, default=None, help="Fallback: single root for both cond/target.")
    parser.add_argument("--data_split", type=str, default='train')
    parser.add_argument("--output_path", type=str, default="path/to/lantent")
    parser.add_argument("--config", type=str, default="tokenizer/configs/vavae_f16d32.yaml")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
