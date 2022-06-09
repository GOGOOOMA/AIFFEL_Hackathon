import os

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from ignite.engine import Engine
from ignite.metrics import FID, PSNR, SSIM
from PIL import Image, ImageDraw, ImageFont
from pytorch_fid.inception import InceptionV3

from dataset import get_real_testset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### metrics (ignite) START ####
class FIDNet(nn.Module):
    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y


def eval_step(engine, batch):
    return batch


default_eval = Engine(eval_step)
psnr = PSNR(data_range=1.0)
psnr.attach(default_eval, "psnr")
ssim = SSIM(data_range=1.0)
ssim.attach(default_eval, "ssim")

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
inception = InceptionV3([block_idx]).to(device)
fid_net = FIDNet(inception)
fid_net.eval()
fid = FID(num_features=2048, feature_extractor=fid_net)
fid.attach(default_eval, "fid")

#### metrics (ignite) END ####


def get_metrics(img1, img2, img3, img4):
    """Get metrics for evaluating the performance
    PSNR, SSIM, FID

    Args:
        img1 (torch.Tensor): synth_output
        img2 (torch.Tensor): synth_gt
        img3 (torch.Tensor): real_output
        img4 (torch.Tensor): real_gt

    Returns:
        tuple: tuple containing:
            psnr_syn
            psnr_real
            ssim_syn
            ssim_real
            fid_syn
            fid_real
    """
    img1, img2, img3, img4 = img1[:32, ::], img2[:32, ::], img3[:32, ::], img4[:32, ::]
    state = default_eval.run([prepare_metric(img1, img2)])
    psnr_syn = state.metrics["psnr"]
    ssim_syn = state.metrics["ssim"]
    fid_syn = state.metrics["fid"]
    state = default_eval.run([prepare_metric(img3, img4)])
    psnr_real = state.metrics["psnr"]
    ssim_real = state.metrics["ssim"]
    fid_real = state.metrics["fid"]
    return psnr_syn, psnr_real, ssim_syn, ssim_real, fid_syn, fid_real


def pick_samples(dataset, device, mode="synth"):
    """Pick samples from dataset

    Args:
        dataset (torch.utils.data.Dataset): _description_
        device (torch.device): _description_
        mode (str, optional): Defaults to "synth".

    Returns:
        tuple: tuple containing:
            data1 (torch.Tensor)
            data2 (torch.Tensor)
    """
    data1 = []
    data2 = []
    if mode == "synth":
        idx_1, idx_2 = 0, 2
    else:  # "real"
        idx_1, idx_2 = 0, 1

    for idx in range(1, 3201, 100):
        data1.append(dataset[idx][idx_1].to(device).unsqueeze(0))
        data2.append(dataset[idx][idx_2].to(device).unsqueeze(0))

    data1 = torch.cat(data1)
    data2 = torch.cat(data2)
    return data1, data2


def pick_samples_test(device):
    """Pick samples from dataset

    Args:
        device (torch.device): _description_

    Returns:
        tuple: tuple containing:
            test_data1 (torch.Tensor)
            test_data2 (torch.Tensor)
    """
    # test image
    test13, test15 = get_real_testset()
    test_data1 = []
    for idx in range(1, 49, 3):
        test_data1.append(test13[idx][1].to(device).unsqueeze(0))
        test_data1.append(test15[idx][1].to(device).unsqueeze(0))
    test_data2 = [make_syn_img().to(device)] * len(test_data1)
    return test_data1, test_data2


@torch.no_grad()
def save_samples(
    enc,
    enc_ct,
    gen,
    test_img1,
    test_img2,
    syn_img1,
    syn_img2,
    real_img1,
    real_img2,
    run_name,
    epoch,
):
    """Save results from sample images

    Args:
        enc (Encoder): _description_
        enc_ct (EncoderCT): _description_
        gen (Generator): _description_
        test_img1 (torch.Tensor): _description_
        test_img2 (torch.Tensor): _description_
        syn_img1 (torch.Tensor): _description_
        syn_img2 (torch.Tensor): _description_
        real_img1 (torch.Tensor): _description_
        real_img2 (torch.Tensor): _description_
        run_name (str): _description_
        epoch (int): _description_
    """

    def get_output(img1, img2):
        real_feat_st, real_feats = enc(img1)
        syn_feat_st, _ = enc(img2)
        syn_feat_ct, _ = enc_ct(syn_feat_st)
        output = gen(real_feat_st, syn_feat_ct.detach(), real_feats)
        return output

    def make_big_grid(img1, img2, output):
        """Make grid image for saving

        Args:
            img1 (torch.Tensor): _description_
            img2 (torch.Tensor): _description_
            output (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        img_input1 = make_grid(img1, nrow=1, normalize=True).permute(1, 2, 0)
        img_input2 = make_grid(img2, nrow=1, normalize=True).permute(1, 2, 0)
        img_output = make_grid(output, nrow=1, normalize=True).permute(1, 2, 0)
        return torch.cat([img_input1, img_input2, img_output], 1).cpu()

    def _make_tensor(img1, img2):
        if not isinstance(img1, torch.Tensor):
            img1 = torch.cat(img1)
            img2 = torch.cat(img2)
        return img1, img2

    os.makedirs(f"result/{run_name}", exist_ok=True)

    enc.eval()
    enc_ct.eval()
    gen.eval()

    fig, axes = plt.subplots(1, 3, dpi=200)

    test_img1, test_img2 = _make_tensor(test_img1, test_img2)
    syn_img1, syn_img2 = _make_tensor(syn_img1, syn_img2)
    real_img1, real_img2 = _make_tensor(real_img1, real_img2)

    # test example
    test_out = get_output(test_img1, test_img2)
    axes[0].imshow(make_big_grid(test_img1, test_img2, test_out))
    axes[0].axis("off")

    batch_size = test_img1.shape[0]
    syn_img1 = syn_img1[:batch_size]
    syn_img2 = syn_img2[:batch_size]
    real_img1 = real_img1[:batch_size]
    real_img2 = real_img2[:batch_size]

    # synth example
    syn_out = get_output(syn_img1, syn_img2)
    axes[1].imshow(make_big_grid(syn_img1, syn_img2, syn_out))
    axes[1].axis("off")

    # real example
    real_out = get_output(real_img1, real_img2)
    axes[2].imshow(make_big_grid(real_img1, real_img2, real_out))
    axes[2].axis("off")

    fig.tight_layout()
    plt.savefig(f"result/{run_name}/{epoch:03d}.png", bbox_inches="tight")
    plt.close()

    enc.train()
    enc_ct.train()
    gen.train()


def prepare_metric(img1, img2, int_mode=False):
    """To use metric calculation function, normalize input images

    Args:
        img1 (torch.Tensor): _description_
        img2 (torch.Tensor): _description_
        int_mode (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: tuple containing:
            img1_new (torch.Tensor): _description_
            img2_new (torch.Tensor): _description_
    """
    img1_new = img1 - img1.min()
    img1_new = img1_new / img1_new.max()
    img2_new = img2 - img2.min()
    img2_new = img2_new / img2_new.max()
    if int_mode:
        return img1_new.type(torch.uint8), img2_new.type(torch.uint8)
    else:
        return img1_new.type(torch.float32), img2_new.type(torch.float32)


def make_syn_img():
    """Make synthetic image for evaluation

    Returns:
        torch.Tensor: syn_img
    """
    img = Image.new("RGB", (128, 32), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size=24
    )
    draw.text((8, 2), "GOGUMA99", fill=(139, 0, 255), font=font)

    transform = transforms.Compose(
        [
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5505, 0.5470, 0.5305), (0.2902, 0.3023, 0.2819)),
        ]
    )
    syn_img = transform(img)
    syn_img = torch.unsqueeze(syn_img, 0)
    return syn_img


def order_ckpt(sample):
    return int(sample.split("-")[0].split("=")[-1])
