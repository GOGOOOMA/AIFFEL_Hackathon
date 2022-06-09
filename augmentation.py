import torch
import torchvision.transforms as transforms


class Augmentor:
    """Class for adaptive discriminator augmentation"""

    def __init__(self, init_p=0.05):
        self.p = torch.Tensor([init_p])

    def brightness(self, x):
        value = torch.Tensor([1]).normal_(std=0.2)
        if value > 0:
            return transforms.functional.adjust_brightness(x, value)
        else:
            return x

    def contrast(self, x):
        temp = torch.log(torch.Tensor([2.0]))[0]
        value = torch.Tensor([1]).log_normal_(std=(0.5 * temp))
        if value > 0:
            return transforms.functional.adjust_contrast(x, value)
        else:
            return x

    def hue(self, x):
        value = (
            torch.Tensor([1]).uniform_(-1 * torch.pi, torch.pi) / (2 * torch.pi)
        ).cuda()
        if value > 0:
            return transforms.functional.adjust_hue(x, value)
        else:
            return x

    def update(self, p):
        self.p = max(0, p)

    def __call__(self, x):
        self.p = self.p.cpu()
        x = transforms.RandomHorizontalFlip(self.p)(x)
        x = transforms.RandomApply([self.brightness, self.contrast, self.hue], self.p)(
            x
        )
        return x
