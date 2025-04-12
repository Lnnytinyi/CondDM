import os
import numpy as np
from PIL import Image
from os.path import join
from torch.utils.data import DataLoader, Dataset

# import torch.multiprocessing as mp
# mp.set_sharing_strategy('file_system')


def list_file_path(data_root, train=True, phase=None):
    """
    get files list for data_root (nct and cct imgs)

    :return data -> lst: nct img path, cct img path, img id.
    """
    data = []
    # if train:
    #     data_dir = join(data_root, "TRAIN")
    # else:
    #     data_dir = join(data_root, "VALID")

    data_dir = join(data_root, phase)

    nct, cct = join(data_dir, "NCT"), join(data_dir, "CCT")
    for img in os.listdir(nct):
        nctimg = join(nct, img)
        cctimg = join(cct, img)

        data.append(
            {
                "nct": nctimg,
                "cct": cctimg,
                "phase": "TRAIN" if train else "VALID",
                "cond": phase,
                "imgid": img,
            }
        )
    return data


def load_data(
    *, data_root, batchsize, imgsize, train=True, phase=None, deterministic=False
):
    """
    load pair nct and cct imgs.

    :param data_root: dataset root dir.
    :param batchsize: data batchsize.
    :param trian: train or valid phase.
    :param phase: venous or artery image.
    :param determinstic: if True, yield results in a deterministic order.
    """
    if not data_root:
        raise ValueError("Unspecified data directory")
    data = list_file_path(data_root, train=train, phase=phase)
    dataset = ImageDataset(data, imgsize=imgsize)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batchsize, shuffle=False, num_workers=16, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batchsize, shuffle=True, num_workers=16, drop_last=True
        )
    # while True:
    #     yield from loader
    return loader


def save_img(*, tensor, save_path, img_name):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    npy = tensor.numpy().squeeze()
    if npy.dtype != np.uint8:
        npy = np.clip((npy + 1) * 127.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(npy, mode="L")
    img.save(join(save_path, img_name))


class ImageDataset(Dataset):
    def __init__(self, imgdict, imgsize):
        super().__init__()
        self.imgdict = imgdict
        self.imgsize = imgsize

    def __len__(self):
        return len(self.imgdict)

    def __getitem__(self, idx):
        """
        :return: a image dict including nct/cct image.
        """
        nctimg = Image.open(self.imgdict[idx]["nct"]).convert("L")
        cctimg = Image.open(self.imgdict[idx]["cct"]).convert("L")
        nctimg.load()
        cctimg.load()

        # resize
        nctimg = nctimg.resize((self.imgsize, self.imgsize), resample=Image.BILINEAR)
        cctimg = cctimg.resize((self.imgsize, self.imgsize), resample=Image.BILINEAR)

        # normalization => [-1 , 1]
        nctimg = np.array(nctimg) / 127.5 - 1
        cctimg = np.array(cctimg) / 127.5 - 1

        # reshape
        nctimg = np.expand_dims(nctimg, axis=0)
        cctimg = np.expand_dims(cctimg, axis=0)

        return {
            "imgid": self.imgdict[idx]["imgid"],
            "nctimg": nctimg.astype(np.float32),
            "cctimg": cctimg.astype(np.float32),
        }


if __name__ == "__main__":
    data_dir = r"/Users/levy/Desktop/DATA/CTtoCTA_Png"
    batchsize = 4
    imgsize = 256

    data = load_data(
        data_root=data_dir,
        batchsize=batchsize,
        imgsize=imgsize,
    )
    for batch in data:
        print(batch["nctimg"].dtype, batch["cctimg"].dtype)
