"""
ImageNet Latent Dataset with safetensors.
This file contains two Dataset classes:
1. ImgLatentDataset: The original class for single-directory datasets, used for feature extraction and stats calculation.
2. PairedImgLatentDataset: A new class for loading paired source/target data for image-to-image training tasks.
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm
import logging

import torch
from torch.utils.data import Dataset

from safetensors import safe_open

logger = logging.getLogger(__name__)


class ImgLatentDataset(Dataset):
    """
    Original ImgLatentDataset for single directory. 
    Used for feature extraction and calculating latent stats.
    """
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()

    def get_img_to_safefile_map(self):
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice('labels')
                labels_shape = labels.get_shape()
                num_imgs = labels_shape[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len+i] = {
                        'safe_file': safe_file,
                        'idx_in_file': i
                    }
        return img_to_file

    def get_latent_stats(self):
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            print(f"Computing latent stats and saving to {latent_stats_cache_file}")
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            # print(f"Loading latent stats from {latent_stats_cache_file}")
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']
    
    def compute_latent_stats(self):
        num_samples = min(10000, len(self.img_to_file_map))
        random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        latents = []
        for idx in tqdm(random_indices, desc="Computing latent stats"):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                features = f.get_slice('latents')
                feature = features[img_idx:img_idx+1]
                latents.append(feature)
        latents = torch.cat(latents, dim=0)
        mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        std = latents.std(dim=[0, 2, 3], keepdim=True)
        latent_stats = {'mean': mean, 'std': std}
        return latent_stats

    def __len__(self):
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            features = f.get_slice(tensor_key)
            labels = f.get_slice('labels')
            feature = features[img_idx:img_idx+1]
            label = labels[img_idx:img_idx+1]

        if self.latent_norm:
            feature = (feature - self._latent_mean) / self._latent_std
        feature = feature * self.latent_multiplier
        
        # remove the first batch dimension (=1) kept by get_slice()
        feature = feature.squeeze(0)
        label = label.squeeze(0)
        return feature, label


class PairedImgLatentDataset(Dataset):
    """
    Dataset for paired source/target data for image-to-image training.
    Assumes latents_stats.pt is pre-computed and exists in both source and target dirs.
    """
    def __init__(self, source_data_dir, target_data_dir, latent_norm=True, latent_multiplier=1.0):
        self.source_data_dir = source_data_dir
        self.target_data_dir = target_data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        self.source_files = sorted(glob(os.path.join(source_data_dir, "*.safetensors")))
        self.target_files = sorted(glob(os.path.join(target_data_dir, "*.safetensors")))
        
        assert len(self.source_files) == len(self.target_files) > 0, "Source and target must have the same number of safetensors files."

        self.img_to_file_map = self.get_img_to_safefile_map()
        
        if latent_norm:
            self._source_latent_mean, self._source_latent_std = self.get_latent_stats(self.source_data_dir)
            self._target_latent_mean, self._target_latent_std = self.get_latent_stats(self.target_data_dir)

    def get_img_to_safefile_map(self):
        img_to_file = {}
        
        source_map = {os.path.basename(f): f for f in self.source_files}

        for target_safe_file in self.target_files:
            basename = os.path.basename(target_safe_file)
            if basename not in source_map:
                logger.warning(f"Target file {basename} not found in source directory. Skipping.")
                continue
            
            source_safe_file = source_map[basename]

            with safe_open(target_safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice('labels')
                labels_shape = labels.get_shape()
                num_imgs = labels_shape[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len + i] = {
                        'source_safe_file': source_safe_file,
                        'target_safe_file': target_safe_file,
                        'idx_in_file': i
                    }
        
        logger.info(f"Found {len(img_to_file)} paired samples.")
        return img_to_file

    def get_latent_stats(self, data_dir):
        latent_stats_cache_file = os.path.join(data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            raise FileNotFoundError(f"Latent stats file not found at {latent_stats_cache_file}. Please run extract_features.py first.")
        else:
            latent_stats = torch.load(latent_stats_cache_file, map_location="cpu")
        return latent_stats['mean'], latent_stats['std']

    def __len__(self):
        return len(self.img_to_file_map)

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        source_safe_file = img_info['source_safe_file']
        target_safe_file = img_info['target_safe_file']
        img_idx = img_info['idx_in_file']

        # Determine whether to use flipped version consistently for both source and target
        use_flip = np.random.uniform(0, 1) > 0.5
        source_tensor_key = "latents_flip" if use_flip else "latents"
        target_tensor_key = "latents_flip" if use_flip else "latents"

        # Load target feature
        with safe_open(target_safe_file, framework="pt", device="cpu") as f:
            target_features_slice = f.get_slice(target_tensor_key)
            labels_slice = f.get_slice('labels')
            target_feature = target_features_slice[img_idx:img_idx+1]
            label = labels_slice[img_idx:img_idx+1]

        # Load source feature
        with safe_open(source_safe_file, framework="pt", device="cpu") as f:
            # Fallback if flip version doesn't exist in source
            try:
                source_features_slice = f.get_slice(source_tensor_key)
            except KeyError:
                source_tensor_key = "latents"
                source_features_slice = f.get_slice(source_tensor_key)
            source_feature = source_features_slice[img_idx:img_idx+1]

        if self.latent_norm:
            target_feature = (target_feature - self._target_latent_mean) / self._target_latent_std
            source_feature = (source_feature - self._source_latent_mean) / self._source_latent_std
        
        target_feature = target_feature * self.latent_multiplier
        source_feature = source_feature * self.latent_multiplier
        
        # Remove the first batch dimension (=1)
        target_feature = target_feature.squeeze(0)
        source_feature = source_feature.squeeze(0)
        label = label.squeeze(0)

        return {
            "image": target_feature,
            "cond": source_feature,
            "class_label": label
        }