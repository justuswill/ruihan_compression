from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ToIntTensor:
    # for IMAGENET64
    def __call__(self, image):
        image = torch.as_tensor(image.reshape(3, 64, 64) / 255.0, dtype=torch.float32)
        return image


class NPZLoader(Dataset):
    """
    Load from a batched numpy datasets.
    Keeps one data batch loaded in memory, so load idx sequentially for fast sampling
    """

    def __init__(self, path, train=True, transform=None):
        self.path = path
        if train:
            self.files = list(Path(path).glob('*train*.npz'))
        else:
            self.files = list(Path(path).glob('*val*.npz'))
        self.batch_lens = [self.npz_len(f) for f in self.files]
        self.anchors = np.cumsum([0] + self.batch_lens)
        self.transform = transform
        self.cache_fid = None
        self.cache_npy = None

    # https://stackoverflow.com/questions/68224572/how-to-determine-the-shape-size-of-npz-file
    @staticmethod
    def npz_len(npz):
        """
        Takes a path to an .npz file, which is a Zip archive of .npy files and returns the batch size of stored data,
        i.e. of the first .npy found
        """
        import zipfile
        with zipfile.ZipFile(npz) as archive:
            for name in archive.namelist():
                if not name.endswith('.npy'):
                    continue
                npy = archive.open(name)
                version = np.lib.format.read_magic(npy)
                shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
                return shape[0]

    def load_npy(self, fid):
        if not fid == self.cache_fid:
            self.cache_fid = fid
            self.cache_npy = np.load(str(self.files[fid]))['data']
        return self.cache_npy

    def __len__(self):
        return sum(self.batch_lens)

    def __getitem__(self, idx):
        fid = np.argmax(idx < self.anchors) - 1
        idx = idx - self.anchors[fid]
        numpy_array = self.load_npy(fid)[idx]
        if self.transform is not None:
            torch_array = self.transform(numpy_array)
        return torch_array[None, :]


def get_data_loaders():
    train_ds = NPZLoader('../datasets/imagenet64', train=True, transform=ToIntTensor())
    eval_ds = NPZLoader('../datasets/imagenet64', train=False, transform=ToIntTensor())
    train = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)
    eval = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=1)
    return train, eval
