import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import utils
from augmentation import Augmentor
from dataset import *

synth_dataset = get_synth_dataset()
real_dataset = get_real_dataset()


def loss_adv(img_gen, img_gt):
    return F.binary_cross_entropy_with_logits(img_gen, img_gt)


def loss_gen(img_gen, img_gt):
    return F.l1_loss(img_gen, img_gt)


def loss_synth_text(decoded, text_label):
    decoded = decoded.view(-1, decoded.shape[-1])
    text_label = text_label[:, 1:].contiguous().view(-1)  # ignore [GO]
    return F.cross_entropy(decoded, text_label, ignore_index=0)


def loss_per(img_gen_fs, img_gt_fs):
    loss = 0.0
    for img_gen_f, img_gt_f in zip(img_gen_fs, img_gt_fs):
        loss += F.l1_loss(img_gen_f, img_gt_f)
    return loss


class Encoder(nn.Module):
    """Encoder using pre-trained ResNet18

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.block1 = nn.Sequential(*modules[:3])
        self.block2 = nn.Sequential(*modules[3:5])
        self.block3 = nn.Sequential(*modules[5])
        self.block4 = nn.Sequential(*modules[6])
        self.block5 = nn.Sequential(*modules[7])

    def forward(self, x):
        f1 = self.block1(x)  # (64, 16, 64)
        f2 = self.block2(f1)  # (64, 8, 32)
        f3 = self.block3(f2)  # (128, 4, 16)
        f4 = self.block4(f3)  # (256, 2, 8)
        f5 = self.block5(f4)  # (512, 1, 4)
        return f5, (f1, f2, f3, f4)


class EncoderCT(nn.Module):
    """Contnet Encoder using BiLSTM

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super(EncoderCT, self).__init__()
        self.content = nn.LSTM(
            2048, 1024, num_layers=1, bidirectional=True, batch_first=True
        )

    def forward(self, x):
        self.content.flatten_parameters()
        content_feat, _ = self.content(x.reshape(-1, 1, 512 * 1 * 4))
        text_feat = content_feat.reshape(-1, 4, 512)
        return content_feat, text_feat


class Generator(nn.Module):
    """Generator based on U-Net architecture(Pix2Pix)

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super(Generator, self).__init__()

        def basic_blk(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        ):
            layers = []
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU())

            block = nn.Sequential(*layers)
            return block

        # Expansive path
        self.dec5_1 = basic_blk(in_channels=1024, out_channels=256)

        self.unpool4 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.dec4_2 = basic_blk(in_channels=512, out_channels=256)
        self.dec4_1 = basic_blk(in_channels=256, out_channels=128)

        self.unpool3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.dec3_2 = basic_blk(in_channels=256, out_channels=128)
        self.dec3_1 = basic_blk(in_channels=128, out_channels=64)

        self.unpool2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.dec2_2 = basic_blk(in_channels=128, out_channels=64)
        self.dec2_1 = basic_blk(in_channels=64, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.dec1_2 = basic_blk(in_channels=128, out_channels=64)
        self.dec1_1 = basic_blk(in_channels=64, out_channels=32)

        self.unpool0 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.final = nn.Conv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, a, b, fs):
        f1, f2, f3, f4 = fs
        x = torch.cat((a, b.reshape(-1, 512, 1, 4)), 1)  # x out (1024, 1, 4)
        dec5_1 = self.dec5_1(x)  # dec5_1 out (256, 1, 4)

        unpool4 = self.unpool4(dec5_1)  # unpool4 out (256, 2, 8)
        cat4 = torch.cat((unpool4, f4), dim=1)  # cat4 out (512, 2, 8)
        dec4_2 = self.dec4_2(cat4)  # 512 -> 256
        dec4_1 = self.dec4_1(dec4_2)  # 256 -> 128

        unpool3 = self.unpool3(dec4_1)  # unpool3 out (128, 4, 16)
        cat3 = torch.cat((unpool3, f3), dim=1)  # cat3 out (256, 4, 16)
        dec3_2 = self.dec3_2(cat3)  # 256 -> 128
        dec3_1 = self.dec3_1(dec3_2)  # 128 -> 64

        unpool2 = self.unpool2(dec3_1)  # unpool2 out (64, 8, 32)
        cat2 = torch.cat((unpool2, f2), dim=1)  # cat2 out (128, 8, 32)
        dec2_2 = self.dec2_2(cat2)  # 128 -> 64
        dec2_1 = self.dec2_1(dec2_2)  # 64 -> 64

        unpool1 = self.unpool1(dec2_1)  # unpool1 out (64, 16, 64)
        cat1 = torch.cat((unpool1, f1), dim=1)  # cat1 out (128, 16, 64)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        unpool0 = self.unpool0(dec1_1)  # unpool0 out (16, 64)
        output = self.final(unpool0)
        return output


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(DisBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class Discriminator(nn.Module):
    """Discriminator based on a discriminator of PatchGAN

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.block1 = DisBlock(in_channels * 2 + 1, 64, normalize=False)
        self.block2 = DisBlock(64, 128)
        self.block3 = DisBlock(128, 256)
        self.block4 = DisBlock(256, 512)
        self.embed = nn.Sequential(*[nn.Linear(512 * 1 * 4, 32 * 128), nn.ReLU()])
        self.patch = nn.Conv2d(512, 1, 3, padding=1)

    def forward(self, a, b, c):
        # batch, channel, height(row), width(col)
        c_re = self.embed(c).reshape(-1, 1, 32, 128)
        x = torch.cat((a, b, c_re), 1)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x = self.patch(x4)
        return x, (x1, x2, x3, x4)


class Recognizer(nn.Module):
    """Modified from https://github.com/clovaai/deep-text-recognition-benchmark

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_size=512, hidden_size=256, n_class=97):
        super(Recognizer, self).__init__()
        self.bilstm = nn.LSTM(input_size, input_size // 2, bidirectional=True)
        self.attention_cell = AttentionCell(input_size, hidden_size, n_class)
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.generator = nn.Linear(hidden_size, n_class)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _char_to_onehot(self, input_char, onehot_dim):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(self.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, feat_ct, text, is_train=True, max_len=15):
        """
        input:
            feat_ct : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_len+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x n_class]
        """
        feat_ct, _ = self.bilstm(feat_ct)
        batch_size = feat_ct.size(0)
        num_steps = max_len + 1  # +1 for [s] at end of sentence.

        output_hiddens = (
            torch.FloatTensor(batch_size, num_steps, self.hidden_size)
            .fill_(0)
            .to(self.device)
        )
        hidden = (
            torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device),
            torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device),
        )

        if is_train:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.n_class)

                # hidden : decoder's hidden s_{t-1}, feat_ct : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, feat_ct, char_onehots)

                # LSTM hidden index (0: hidden, 1: Cell)
                output_hiddens[:, i, :] = hidden[0]
            probs = self.generator(output_hiddens)

        else:
            targets = (
                torch.LongTensor(batch_size).fill_(0).to(self.device)
            )  # [GO] token
            probs = (
                torch.FloatTensor(batch_size, num_steps, self.n_class)
                .fill_(0)
                .to(self.device)
            )

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.n_class)
                hidden, alpha = self.attention_cell(hidden, feat_ct, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x n_class


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        # either i2i or h2h should have bias
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, feat_ct, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]

        # feat_ct (N, 4, 512), feat_ct_proj (N, 4, 256)
        feat_ct_proj = self.i2h(feat_ct)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)

        # batch_size x num_encoder_step * 1
        e = self.score(torch.tanh(feat_ct_proj + prev_hidden_proj))
        alpha = F.softmax(e, dim=1)  # alpha (N, 4, 1)

        # batch_size x num_channel
        context = torch.bmm(alpha.permute(0, 2, 1), feat_ct).squeeze(1)

        # batch_size x (num_channel + num_embedding)
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


