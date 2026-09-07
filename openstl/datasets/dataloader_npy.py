"""Generic dataloader for per-sample .npy files.

Expected directory layout::

    data_root/
    ├── train/    # one .npy file per sample
    ├── val/      # one .npy file per sample
    └── test/     # one .npy file per sample

Each .npy file holds one sample of shape ``(T, C, H, W)`` where
``T = pre_seq_length + aft_seq_length``.  ``(T, H, W)`` grayscale and
``(T, H, W, C)`` channels-last layouts are auto-detected.

Usage::

    python tools/train.py --dataname npy --data_root data ...

Optional ``--data_name``-style knobs are passed through the training
config; see ``load_data`` below.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class NpySequenceDataset(Dataset):
    """One .npy file per video sample, loaded on demand.

    Args:
        data_root (str): Root directory containing train/val/test subdirs.
        split (str): One of 'train', 'val', 'test'.
        pre_seq_length (int): Number of input frames.
        aft_seq_length (int): Number of frames to predict.
        channel_last (bool|None): Force channels-last layout.  None =
            auto-detect from shape.
        normalize (bool): Divide uint8 data by 255.
    """

    def __init__(self, data_root, split='train',
                 pre_seq_length=10, aft_seq_length=10,
                 channel_last=None, normalize=True):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_root, split, '*.npy')))
        assert len(self.files) > 0, \
            f'No .npy files found in {os.path.join(data_root, split)}'
        self.pre_seq = pre_seq_length
        self.aft_seq = aft_seq_length
        self.total = pre_seq_length + aft_seq_length
        self.channel_last = channel_last
        self.normalize = normalize
        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.files)

    def _to_tcw(self, arr):
        """Normalize any supported layout to (T, C, H, W)."""
        if arr.ndim == 3:  # (T, H, W) grayscale
            return arr[:, np.newaxis, ...]
        if arr.ndim == 4:
            # Heuristic: channels are tiny (1-16), H/W usually larger.
            if self.channel_last is None:
                channel_last = (arr.shape[-1] <= 16 and arr.shape[1] > 16)
            else:
                channel_last = self.channel_last
            if channel_last:
                return arr.transpose(0, 3, 1, 2)
            return arr
        raise ValueError(f'Unsupported sample shape {arr.shape} (need 3D or 4D)')

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        arr = self._to_tcw(arr)
        assert arr.shape[0] >= self.total, \
            f'{self.files[idx]}: {arr.shape[0]} frames < required {self.total}'

        if arr.dtype == np.uint8 and self.normalize:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)

        seq = torch.from_numpy(arr).contiguous()
        input_frames = seq[:self.pre_seq]
        output_frames = seq[self.pre_seq:self.total]
        return input_frames, output_frames


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=10, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False,
              drop_last=False, **kwargs):
    """Build train/val/test loaders from per-sample .npy directories."""

    channel_last = kwargs.get('channel_last', None)
    normalize = kwargs.get('normalize', True)

    train_set = NpySequenceDataset(
        data_root, split='train',
        pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
        channel_last=channel_last, normalize=normalize)
    val_set = NpySequenceDataset(
        data_root, split='val',
        pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
        channel_last=channel_last, normalize=normalize)
    test_set = NpySequenceDataset(
        data_root, split='test',
        pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
        channel_last=channel_last, normalize=normalize)

    common = dict(pin_memory=True, drop_last=drop_last,
                  num_workers=num_workers, distributed=distributed,
                  use_prefetcher=use_prefetcher)

    dataloader_train = create_loader(train_set, batch_size=batch_size,
                                     shuffle=True, is_training=True, **common)
    dataloader_vali = create_loader(val_set, batch_size=val_batch_size,
                                    shuffle=False, is_training=False, **common)
    dataloader_test = create_loader(test_set, batch_size=val_batch_size,
                                    shuffle=False, is_training=False, **common)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=4, val_batch_size=2, data_root='../../data/npy',
                  num_workers=0, pre_seq_length=10, aft_seq_length=10)
    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
