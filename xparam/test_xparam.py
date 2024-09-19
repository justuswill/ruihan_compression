import argparse
import os

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.compress_modules import ResnetCompressor
from ema_pytorch import EMA
from itertools import islice

# from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
# from pytorch_fid.inception import InceptionV3
from torch_fidelity import register_dataset, calculate_metrics
from xparam.data_loader import get_data_loaders

parser = argparse.ArgumentParser(description="values from bash script")

parser.add_argument("--ckpt", type=str, required=True)  # ckpt path
parser.add_argument("--gamma", type=float, default=0.8)  # noise intensity for decoding
parser.add_argument("--n_denoise_step", type=int, default=65)  # number of denoising step
parser.add_argument("--device", type=int, default=0)  # gpu device index
parser.add_argument("--img_dir", type=str, default='../imgs')
parser.add_argument("--out_dir", type=str, default='../compressed_imgs')
parser.add_argument("--lpips_weight", type=float, required=True)  # either 0.9 or 0.0, note that this must match the ckpt you use, because with weight>0, the lpips-vggnet weights were also saved during training. Incorrect state_dict keys may lead to load_state_dict error when loading the ckpt.

config = parser.parse_args()


def main(rank):
    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=64,
        dim_mults=[1, 2, 3, 4, 5, 6],
        context_dim_mults=[1, 2, 3, 4],
        embd_type="01",
    )

    context_model = ResnetCompressor(
        dim=64,
        dim_mults=[1, 2, 3, 4],
        reverse_dim_mults=[4, 3, 2, 1],
        hyper_dims_mults=[4, 4, 4],
        channels=3,
        out_channels=64,
    )

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        ae_fn=None,
        num_timesteps=8193,
        # num_timesteps=20000,
        loss_type="l2",
        lagrangian=0.0032,
        pred_mode="x",
        aux_loss_weight=config.lpips_weight,
        aux_loss_type="lpips",
        var_schedule="cosine",
        use_loss_weight=True,
        loss_weight_min=5,
        use_aux_loss_weight_schedule=False,
    )
    loaded_param = torch.load(
        config.ckpt,
        map_location=lambda storage, loc: storage,
    )
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
    ema.load_state_dict(loaded_param["ema"])
    diffusion = ema.ema_model
    diffusion.to(rank)
    diffusion.eval()

    train, eval = get_data_loaders()

    # for img in os.listdir(config.img_dir):
    #     if img.endswith(".png") or img.endswith(".jpg"):
    #         to_be_compressed = torchvision.io.read_image(os.path.join(config.img_dir, img)).unsqueeze(0).float().to(
    #             rank) / 255.0
    #         compressed, bpp = diffusion.compress(
    #             to_be_compressed * 2.0 - 1.0,
    #             sample_steps=config.n_denoise_step,
    #             bpp_return_mean=True,
    #             init=torch.randn_like(to_be_compressed) * config.gamma
    #         )
    #         compressed = compressed.clamp(-1, 1) / 2.0 + 0.5
    #         pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)
    #         torchvision.utils.save_image(compressed.cpu(), os.path.join(config.out_dir, img))
    #         print("bpp:", bpp)

    # legacy
    train = iter(train)
    for _ in range(8):
        next(train)

    with torch.no_grad():
        bpps = []
        psnrs = []
        img_i = 0
        for batch in islice(train, 1250):
            x = batch.to(rank)[0]
            x_hat, bpp = diffusion.compress(
                x * 2 - 1,
                sample_steps=config.n_denoise_step,
                bpp_return_mean=True,
                init=torch.randn_like(x) * config.gamma
            )
            x_hat = x_hat.clamp(-1, 1) / 2.0 + 0.5
            # save to png
            for j in range(x_hat.shape[0]):
                if img_i == 10000:
                    break
                im = Image.fromarray(np.round(np.transpose(x_hat[j].cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8))
                save_dir = '../imgs/%s' % os.path.basename(config.ckpt).split('-b')[1]
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                im.save(save_dir + '/%d.png' % img_i, 'png')
                img_i += 1
            if img_i == 10000:
                break
            # fid = calculate_metrics(input1='imagenet64', input2=DummyDataset(x_hat), cuda=True, fid=True, verbose=False)['frechet_inception_distance']
            psnr = -10 * np.log10(((x_hat - x) ** 2).mean().item()) + 20 * np.log10(1)
            bpps += [bpp.item()]
            psnrs += [psnr]
    bpp = sum(bpps) / len(bpps)
    psnr = sum(psnrs) / len(psnrs)
    print("bpp:", bpp)
    print("psnr:", psnr)
    return bpp, psnr


# def fid_stats():
#     path = '../datasets/imagenet64_stats'
#     if os.path.exists(path):
#         saved = np.load(path)
#         return saved['mu'], saved['sigma']
#     else:
#         print('Computing stats for evaluation set')
#         model = InceptionV3([3]).to('cpu')
#         model.eval()
#         _, eval = get_data_loaders()
#         mu, sigma = calculate_activation_statistics(eval, model, batch_size=512, dims=2048, device='cpu')
#         np.savez_compressed(path, mu=mu, sigma=sigma)
#         return mu, sigma
#
#
# def compute_fid(batch):
#     m1, s1 = fid_stats()
#     model = InceptionV3([3]).to('cpu')
#     model.eval()
#     m2, s2 = calculate_activation_statistics(batch, model, batch_size=512, dims=2048, device='cpu')
#     return calculate_frechet_distance(m1, s1, m2, s2)


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    bpps = []
    psnrs = []
    # for ckpt in ('../checkpoints/image-l2-use_weight5-imagenet64-d64-t8193-b0.0128-x-cosine-01-float32-aux0.0/image-l2-use_weight5-imagenet64-d64-t8193-b0.0128-x-cosine-01-float32-aux0.0_1.pt', ):
    # for ckpt in ('../checkpoints/image-l2-use_weight5-vimeo-d64-t8193-b0.1024-x-cosine-01-float32-aux0.9lpips_2.pt', ):
    for ckpt in os.listdir('../checkpoints'):
        if '.pt' not in ckpt:
            continue
        config.ckpt = '../checkpoints/' + ckpt
        config.lpips_weight = float(ckpt.split('aux')[1][:3])
        # print(float(ckpt.split('-b')[1].split('-x-')[0]), config.lpips_weight)
        bpp, psnr = main(config.device)
        bpps += [bpp]
        psnrs += [psnr]

    # ordered ckpts
    bpps = np.array(bpps)[[8, 7, 6, 5, 4, 3, 0, 2, 1]]
    psnrs = np.array(psnrs)[[8, 7, 6, 5, 4, 3, 0, 2, 1]]
    np.savez_compressed('../results/cdc.npz', psnrs=psnrs, bpps=bpps)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(bpps[5:], psnrs[5:])
    ax.plot(bpps[5:], psnrs[5:])
    ax.set(xlabel='bpp', ylabel='psnr', title='bpp vs psnr')
    ax.grid()
    fig.savefig("../results/bpp_vs_psnr.png")
    plt.show()
