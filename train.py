import argparse
import os
import random
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from augmentation import Augmentor
from dataset import *
from model import *

manualSeed = 22603
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
cudnn.deterministic = True
cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1500)
parser.add_argument("--batch", type=int, default=384)
parser.add_argument("--max_lr", type=float, default=1e-3)
parser.add_argument("--init_lr", type=float, default=1e-4)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--scheduler", type=str, default="cyclic")
parser.add_argument("--ada", action="store_true")
parser.add_argument("--version", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--perw", type=float, default=2.0)
parser.add_argument("--advw", type=float, default=0.5)
args = parser.parse_args()

args.ada = True

synth_dataset = torch.load("dataset/synth_187k.pt")
synth_dataloader = DataLoader(
    synth_dataset,
    batch_size=args.batch * 3 // 4,
    shuffle=True,
    num_workers=args.workers,
    # pin_memory=True,
)
print()
print("Synth Dataset Loaded")

real_dataset = torch.load("dataset/real_dataset.pt")
real_dataloader = DataLoader(
    real_dataset,
    batch_size=args.batch // 4,
    shuffle=True,
    num_workers=args.workers,
    # pin_memory=True,
)
print("Real Dataset Loaded")

print()
print(" ==== TT ==== ")
print("Epoch      :", args.epoch)
print("Batch Size :", args.batch)
print("Init LR    :", args.init_lr)
print("Max LR     :", args.max_lr)
print("Optimizer  :", args.optim)
print("Scheduler  :", args.scheduler)
print("ADA        :", args.ada)
print("Per W      :", args.perw)
print("Adv W      :", args.advw)
print("Version    :", args.version)
print("Resume     :", args.resume)
print("Device     :", device)
print("Synth Data :", len(synth_dataset))
print("Real Data  :", len(real_dataset))
print()

syn_test_data1, syn_test_data2 = utils.pick_samples(synth_dataset, device, mode="synth")
real_test_data1, real_test_data2 = utils.pick_samples(real_dataset, device, mode="real")
test_data1, test_data2 = utils.pick_samples_test(device)

n_class = len(LabelConverterAtt().character)

E = Encoder().to(device)
ECT = EncoderCT().to(device)
G = Generator().to(device)
D = Discriminator().to(device)
R = Recognizer(n_class=n_class, device=device).to(device)

init_lr = args.init_lr
if args.optim == "adam":
    opt_enc = optim.Adam(E.parameters(), lr=init_lr)
    opt_enc_ct = optim.Adam(ECT.parameters(), lr=init_lr)
    opt_gen = optim.Adam(G.parameters(), lr=init_lr)
    opt_dis = optim.Adam(D.parameters(), lr=init_lr)
    opt_rec = optim.Adam(R.parameters(), lr=init_lr)
else:  # adam_w
    opt_enc = optim.AdamW(E.parameters(), lr=init_lr)
    opt_enc_ct = optim.AdamW(ECT.parameters(), lr=init_lr)
    opt_gen = optim.AdamW(G.parameters(), lr=init_lr)
    opt_dis = optim.AdamW(D.parameters(), lr=init_lr)
    opt_rec = optim.AdamW(R.parameters(), lr=init_lr)

if args.scheduler == "cyclic":
    scheduler_dict = {
        "base_lr": 1e-4,
        "max_lr": args.max_lr,
        "step_size_up": 3e5,
        "cycle_momentum": False,
    }
    sch_enc = optim.lr_scheduler.CyclicLR(opt_enc, **scheduler_dict)
    sch_enc_ct = optim.lr_scheduler.CyclicLR(opt_enc_ct, **scheduler_dict)
    sch_gen = optim.lr_scheduler.CyclicLR(opt_gen, **scheduler_dict)
    sch_dis = optim.lr_scheduler.CyclicLR(opt_dis, **scheduler_dict)
    sch_rec = optim.lr_scheduler.CyclicLR(opt_rec, **scheduler_dict)
else:  # cosine annealing
    scheduler_dict = {
        "T_max": 3e5,
    }
    sch_enc = optim.lr_scheduler.CosineAnnealingLR(opt_enc, **scheduler_dict)
    sch_enc_ct = optim.lr_scheduler.CosineAnnealingLR(opt_enc_ct, **scheduler_dict)
    sch_gen = optim.lr_scheduler.CosineAnnealingLR(opt_gen, **scheduler_dict)
    sch_dis = optim.lr_scheduler.CosineAnnealingLR(opt_dis, **scheduler_dict)
    sch_rec = optim.lr_scheduler.CosineAnnealingLR(opt_rec, **scheduler_dict)

train_start_t = datetime.now(timezone(timedelta(hours=9)))
print("TRAIN START:", train_start_t)
print()

run_name = f"perw_{args.perw}_advw_{args.advw}_batch_{args.batch}_epoch_{args.epoch}_v{args.version}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
os.makedirs(f"result/{run_name}", exist_ok=True)

aug_synth = Augmentor(device)
aug_real = Augmentor(device)

if not args.resume:
    epoch_start = 0
else:  # resume == True
    ckpt_path = sorted(
        glob.glob(f"weights/{run_name}/ckpt_*.pt"),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )[-1]
    checkpoint = torch.load(ckpt_path)
    epoch_start = checkpoint["epoch"]
    E.load_state_dict(checkpoint["E"])
    ECT.load_state_dict(checkpoint["Ect"])
    G.load_state_dict(checkpoint["G"])
    D.load_state_dict(checkpoint["D"])
    R.load_state_dict(checkpoint["R"])
    opt_enc.load_state_dict(checkpoint["opt_enc"])
    opt_enc_ct.load_state_dict(checkpoint["opt_enc_ct"])
    opt_gen.load_state_dict(checkpoint["opt_gen"])
    opt_dis.load_state_dict(checkpoint["opt_dis"])
    opt_rec.load_state_dict(checkpoint["opt_rec"])
    sch_enc.load_state_dict(checkpoint["sch_enc"])
    sch_enc_ct.load_state_dict(checkpoint["sch_enc_ct"])
    sch_gen.load_state_dict(checkpoint["sch_gen"])
    sch_dis.load_state_dict(checkpoint["sch_dis"])
    sch_rec.load_state_dict(checkpoint["sch_rec"])
    aug_synth.update(checkpoint["aug_synth_p"])
    aug_real.update(checkpoint["aug_real_p"])

# model save
if not os.path.isdir(f"weights/{run_name}"):
    os.mkdir(f"weights/{run_name}")

scaler = torch.cuda.amp.GradScaler()
steps = min(len(synth_dataloader), len(real_dataloader))

for epoch in range(epoch_start, args.epoch):
    start_t = time.time()
    for idx, (synth, real) in enumerate(zip(synth_dataloader, real_dataloader)):
        # input data
        syn1_img, syn1_label, syn2_img, _, _ = synth
        syn1_img, syn1_label, syn2_img = (
            syn1_img.to(device),
            syn1_label.to(device),
            syn2_img.to(device),
        )
        real1_img, real2_img = real
        real1_img, real2_img = real1_img.to(device), real2_img.to(device)

        with torch.cuda.amp.autocast():
            # encoder, generator, recognizer, discriminator forward
            syn1_st, syn1_feats = E(syn1_img)
            syn2_st, _ = E(syn2_img)
            _, syn1_txt = ECT(syn1_st)
            syn2_ct, _ = ECT(syn2_st)
            syn1_decoded = R(syn1_txt, syn1_label[:, :-1])
            syn_output = G(syn1_st, syn2_ct.detach(), syn1_feats)
            syn1_pred, syn1_fs = D(syn2_img, syn1_img, syn2_ct.detach())
            syn2_pred, syn2_fs = D(syn_output, syn1_img, syn2_ct.detach())

            real1_st, real1_feats = E(real1_img)
            real2_st, _ = E(real2_img)
            real2_ct, _ = ECT(real2_st)
            real_output = G(real1_st, real2_ct.detach(), real1_feats)
            real1_pred, real1_fs = D(real2_img, real1_img, real2_ct.detach())
            real2_pred, real2_fs = D(real_output, real1_img, real2_ct.detach())

            # encoder, generator, recognizer backward
            l_gen_syn = loss_gen(syn_output, syn2_img)
            l_gen_real = loss_gen(real_output, real2_img)
            l_per_syn = loss_per(syn1_fs, syn2_fs)
            l_per_real = loss_per(real1_fs, real2_fs)

            loss_R = loss_synth_text(syn1_decoded, syn1_label)

            # gt synth, generated synth
            l_adv_syn1 = loss_adv(syn1_pred, torch.ones_like(syn1_pred))
            l_adv_syn2 = loss_adv(syn2_pred, torch.zeros_like(syn2_pred))
            # gt real, generated real
            l_adv_real1 = loss_adv(real1_pred, torch.ones_like(real1_pred))
            l_adv_real2 = loss_adv(real2_pred, torch.zeros_like(real2_pred))
            l_adv_syn = l_adv_syn1 + l_adv_syn2
            l_adv_real = l_adv_real1 + l_adv_real2
            loss_D = args.advw * (l_adv_syn + l_adv_real)

            loss_EG = (
                l_gen_syn
                + l_gen_real
                + args.perw * l_per_syn
                + args.perw * l_per_real
                + loss_D
            )

        opt_enc.zero_grad()
        opt_gen.zero_grad()
        opt_enc_ct.zero_grad()
        opt_rec.zero_grad()
        scaler.scale(loss_EG).backward(retain_graph=True)
        scaler.scale(loss_R).backward()
        scaler.step(opt_enc)
        scaler.step(opt_gen)
        scaler.step(opt_enc_ct)
        scaler.step(opt_rec)

        with torch.cuda.amp.autocast():
            # discriminator forward
            with torch.no_grad():
                syn1_st, syn1_feats = E(syn1_img)
                syn2_st, syn2_feats = E(syn2_img)
                syn2_ct, _ = ECT(syn2_st)
                syn_output = G(syn1_st, syn2_ct.detach(), syn1_feats)

                real1_st, real1_feats = E(real1_img)
                real2_st, real2_feats = E(real2_img)
                real2_ct, _ = ECT(real2_st)
                real_output = G(real1_st, real2_ct.detach(), real1_feats)

            if args.ada:
                # discriminator augmentation
                syn1_img = aug_synth(syn1_img)
                syn2_img = aug_synth(syn2_img)
                syn_output = aug_synth(syn_output)
                real1_img = aug_real(real1_img)
                real2_img = aug_real(real2_img)
                real_output = aug_real(real_output)

            syn1_pred, syn1_fs = D(syn2_img, syn1_img, syn2_ct.detach())
            syn2_pred, syn2_fs = D(syn_output, syn1_img, syn2_ct.detach())
            real1_pred, real1_fs = D(real2_img, real1_img, real2_ct.detach())
            real2_pred, real2_fs = D(real_output, real1_img, real2_ct.detach())

            # discriminator backward
            l_adv_syn1 = loss_adv(syn1_pred, torch.ones_like(syn1_pred))  # gt
            l_adv_syn2 = loss_adv(syn2_pred, torch.zeros_like(syn2_pred))  # generated
            l_adv_real1 = loss_adv(real1_pred, torch.ones_like(real1_pred))
            l_adv_real2 = loss_adv(real2_pred, torch.zeros_like(real2_pred))
            l_adv_syn = l_adv_syn1 + l_adv_syn2
            l_adv_real = l_adv_real1 + l_adv_real2
            loss_D = 1 * (l_adv_syn + l_adv_real)

            opt_dis.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(opt_dis)

        scaler.update()

        # lr scheduler
        sch_enc.step()
        sch_enc_ct.step()
        sch_gen.step()
        sch_dis.step()
        sch_rec.step()

        elapsed = time.time() - start_t
        if (idx + 1) % 8 == 0:
            train_log = f"EPOCH {epoch+1:04d} | Elapsed: {int(elapsed):3d} | LR: {sch_enc.get_last_lr()[0]:.6f} | Loss EG: {loss_EG.item():.3f} | Loss D: {loss_D.item():.3f} | Loss R: {loss_R.item():.3f}"
            print(train_log, end="\r", flush=True)
        if args.ada and (idx + 1) % 4 == 0:
            # adaptive disriminator augmentation probabilty update
            p_synth_adjust = torch.sign(syn1_pred.detach()).mean() * 6e-4 / steps
            p_real_adjust = torch.sign(real1_pred.detach()).mean() * 6e-4 / steps
            aug_synth.update(p=aug_synth.p + p_synth_adjust.cpu())
            aug_real.update(p=aug_real.p + p_real_adjust.cpu())

    # end of epoch
    train_log = f"EPOCH {epoch+1:04d} | Elapsed: {int(elapsed):3d} | LR: {sch_enc.get_last_lr()[0]:.6f} | Loss EG: {loss_EG.item():.3f} | Loss D: {loss_D.item():.3f} | Loss R: {loss_R.item():.3f}"
    print(train_log)
    writer.add_scalar("EGR/loss_syn_gen", l_gen_syn, epoch)
    writer.add_scalar("EGR/loss_real_gen", l_gen_real, epoch)
    writer.add_scalar("EGR/loss_syn_per", l_per_syn, epoch)
    writer.add_scalar("EGR/loss_real_per", l_per_real, epoch)
    writer.add_scalar("EGR/loss_EG", loss_EG, epoch)
    writer.add_scalar("EGR/loss_R", loss_R, epoch)
    writer.add_scalar("EGR/learning_rate", sch_enc.get_last_lr()[0], epoch)

    writer.add_scalar("D/loss_syn_a", l_adv_syn, epoch)
    writer.add_scalar("D/loss_real_a", l_adv_real, epoch)
    writer.add_scalar("D/loss_d", loss_D, epoch)

    # discriminator synth acc
    syn_gen_acc = torch.mean(torch.ge(syn1_pred.detach(), 0.5).type(torch.FloatTensor))
    syn_gt_acc = torch.mean(torch.less(syn2_pred.detach(), 0.5).type(torch.FloatTensor))

    # discriminator real acc
    real_gen_acc = torch.mean(
        torch.ge(real1_pred.detach(), 0.5).type(torch.FloatTensor)
    )
    real_gt_acc = torch.mean(
        torch.less(real2_pred.detach(), 0.5).type(torch.FloatTensor)
    )

    writer.add_scalar("D/syn_gen_acc", syn_gen_acc, epoch)
    writer.add_scalar("D/syn_gt_acc", syn_gt_acc, epoch)
    writer.add_scalar("D/real_gen_acc", real_gen_acc, epoch)
    writer.add_scalar("D/real_gt_acc", real_gt_acc, epoch)

    writer.add_scalar("D/prob_syn", aug_synth.p, epoch)
    writer.add_scalar("D/prob_real", aug_real.p, epoch)

    if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epoch:
        (
            psnr_syn,
            psnr_real,
            ssim_syn,
            ssim_real,
            fid_syn,
            fid_real,
        ) = utils.get_metrics(syn_output, syn2_img, real_output, real2_img)
        writer.add_scalar("Metrics/psnr_synth", psnr_syn, epoch)
        writer.add_scalar("Metrics/psnr_real", psnr_real, epoch)
        writer.add_scalar("Metrics/ssim_synth", ssim_syn, epoch)
        writer.add_scalar("Metrics/ssim_real", ssim_real, epoch)
        writer.add_scalar("Metrics/fid_synth", fid_syn, epoch)
        writer.add_scalar("Metrics/fid_real", fid_real, epoch)

    if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epoch:
        # test
        utils.save_samples(
            E,
            ECT,
            G,
            test_data1,
            test_data2,
            syn_test_data1,
            syn_test_data2,
            real_test_data1,
            real_test_data2,
            run_name,
            epoch,
        )

    if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epoch:
        torch.save(
            {
                "epoch": epoch,
                "E": E.state_dict(),
                "Ect": ECT.state_dict(),
                "G": G.state_dict(),
                "D": D.state_dict(),
                "R": R.state_dict(),
                "opt_enc": opt_enc.state_dict(),
                "opt_enc_ct": opt_enc_ct.state_dict(),
                "opt_gen": opt_gen.state_dict(),
                "opt_dis": opt_dis.state_dict(),
                "opt_rec": opt_rec.state_dict(),
                "sch_enc": sch_enc.state_dict(),
                "sch_enc_ct": sch_enc_ct.state_dict(),
                "sch_gen": sch_gen.state_dict(),
                "sch_dis": sch_dis.state_dict(),
                "sch_rec": sch_rec.state_dict(),
                "aug_synth_p": aug_synth.p,
                "aug_real_p": aug_real.p,
            },
            f"weights/{run_name}/ckpt_{epoch}.pt",
        )

    if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epoch:
        torch.save(E.state_dict(), f"weights/{run_name}/enc_{epoch}.pt")
        torch.save(ECT.state_dict(), f"weights/{run_name}/enc_ct_{epoch}.pt")
        torch.save(G.state_dict(), f"weights/{run_name}/gen_{epoch}.pt")

torch.save(E.state_dict(), f"weights/{run_name}/enc_final.pt")
torch.save(ECT.state_dict(), f"weights/{run_name}/enc_ct_final.pt")
torch.save(G.state_dict(), f"weights/{run_name}/gen_final.pt")

writer.flush()
print()
train_end_t = datetime.now(timezone(timedelta(hours=9)))
print(f"TRAIN END  : {train_end_t}")
elapsed_t = train_end_t - train_start_t
print(f"ELAPSED    : {elapsed_t}")
writer.close()
