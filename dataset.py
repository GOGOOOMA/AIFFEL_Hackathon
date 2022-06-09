import glob
import os
import string

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm


def ic1517_to_mine(coords):
    """Convert IC15, IC17 coordinates to My Coordinates

    Args:
        coords (list): _description_

    Returns:
        tuple: Bounding Rectangle
    """
    # Convert IC15(IC17) coordinates to My coordinates
    xs = [int(coords[i]) for i in range(0, len(coords), 2)]  # even -> xs
    ys = [int(coords[i]) for i in range(1, len(coords), 2)]  # odd -> ys
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    return ((x_min, y_min), (x_max, y_max))


def coco_to_mine(coords):
    """Convert COCO coordinates to My Coordinates

    Args:
        coords (list): _description_

    Returns:
        tuple: Bounding Rectangle
    """
    x, y, w, h = map(int, coords)
    return ((x, y), (x + w, y + h))


def process_ic13_label(label):
    """Process IC13 label data

    Coordinates are divided by ' '(white space)
    Consist of 4 numbers(top-left, bottom-right)

    Args:
        label (list): list of coordinates

    Returns:
        list: list containing:
            tuple: (coords, text)
    """
    new_label = []
    for l in label:
        info = l.split()
        coords = list(map(int, info[:4]))
        new_coords = ((coords[0], coords[1]), (coords[2], coords[3]))
        new_label.append((new_coords, info[-1][1:-1]))
    # (coords, text), (coords, text), ...
    return new_label


def process_ic15_label(label):
    """Process IC15 label data

    Coordinates are divided by ','
    Consist of 8 numbers(top-left, top-right, bottom-right, bottom-left)
    Args:
        label (list): _description_

    Returns:
        list: list containing:
            tuple: (coords, text)
    """
    new_label = []
    for l in label:
        info = l.split(",")
        if info[-1] != "###\n":
            coords = ic1517_to_mine(info[:8])
            new_label.append((coords, info[-1][:-1]))
    return new_label


def process_ic1719_label(label, lang):
    """Process IC17, IC19 label data

    Args:
        label (_type_): _description_
        lang (str): _description_

    Returns:
        list: list containing:
            tuple: (coords, text)
    """
    new_label = []
    for l in label:
        info = l.split(",")
        if info[-1] != "###\n" and info[8] in lang:
            coords = ic1517_to_mine(info[:8])
            new_label.append((coords, info[-1][:-1]))
    return new_label


def process_coco_label(label):
    """Process COCO-Text label data

    Coodinates are in a form of python list
    Consist of 4 numbers(left, top, width, height)

    Args:
        label (list): _description_

    Returns:
        list: list containing:
            tuple: (coords, text)
    """

    new_label = []
    for l in label:
        if l["legibility"] == "legible":
            bbox = l["bbox"]
            coords = coco_to_mine(bbox)
            text = l["utf8_string"]
            new_label.append((coords, text))
    return new_label


def read_label_IC(label_path, tag, lang):
    """Read label data from IC13, IC15, IC17, IC19

    Args:
        label_path (str): _description_
        tag (str): _description_
        lang (str): _description_

    Returns:
        list: list containing:
            tuple: (coords, text)
    """
    with open(label_path, "r", encoding="utf-8-sig") as f:
        label = f.readlines()
    if tag == "IC13":
        label = process_ic13_label(label)
    elif tag == "IC15":
        label = process_ic15_label(label)
    else:  # IC17, IC19
        label = process_ic1719_label(label, lang)
    return label


def cut_out(img, length=42):
    """Cut out data augmentation with fixed length

    Args:
        img (torch.Tensor): _description_
        length (int, optional): _description_. Defaults to 42.

    Returns:
        torch.Tensor: img with cut out
    """
    h = img.size(1)
    w = img.size(2)
    x = np.random.randint(w)

    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask = np.ones((h, w), np.float32)
    mask[0:h, x1:x2] = 0.0
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask
    return img


transform_pil = transforms.ToPILImage()
transform_tensor = transforms.ToTensor()


def crop_image(img, labels):
    """Crop the image to size 128x32

    Args:
        img (_type_): _description_
        labels (_type_): _description_

    Returns:
        list: list containing:
            torch.Tensor: cropped image
            str: text
    """
    boxes = []
    img = transform_pil(img)
    for label in labels:
        img_orig = img.copy()
        coords, text = label
        img_crop = img_orig.crop(
            (coords[0][0], coords[0][1], coords[1][0], coords[1][1])
        )
        img_resize = img_crop.resize((128, 32))
        img_tensor = transform_tensor(img_resize)
        boxes.append((img_tensor, text))
    return boxes


class ICDatasetCut(Dataset):
    """Dataset for IC13, IC15, IC17, IC19

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, img_dir, label_dir, transform=None, tag="IC13", lang=["Latin"]):
        self.img_path_list = sorted(glob.glob(img_dir + "/*"))
        self.label_path_list = sorted(glob.glob(label_dir + "/*"))
        self.transform = transform
        self.tag = tag
        self.lang = lang

        self.images = []
        self.labels = []
        self.boxes = []
        self.invalid = 0

        for img_path, label_path in tqdm(
            zip(self.img_path_list, self.label_path_list),
            total=len(self.label_path_list),
        ):
            label = read_label_IC(label_path, self.tag, self.lang)
            self.labels.append(label)
            if label:
                try:
                    image = read_image(img_path)
                except Exception as err:
                    # print(img_path)
                    self.invalid += 1
                boxes = crop_image(image, label)
                self.boxes.extend(boxes)

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        image, text = self.boxes[idx]
        image_cut = cut_out(image)
        if self.transform:
            image = self.transform(image)
            image_cut = self.transform(image_cut)
        return image_cut, image


class ICDataset(Dataset):
    """Dataset for IC13, IC15, IC17, IC19

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, img_dir, transform=None, tag="IC13"):
        self.box_path_list = glob.glob(img_dir + f"/{tag}_*.jpg")[::3]
        self.transform = transform

    def __len__(self):
        return len(self.box_path_list)

    def __getitem__(self, idx):
        image = read_image(self.box_path_list[idx]) / 255.0
        image_cut = cut_out(image)
        if self.transform:
            image = self.transform(image)
            image_cut = self.transform(image_cut)
        return image_cut, image


class COCODataset(Dataset):
    """Dataset for COCO-Text

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, img_dir, transform=None):
        self.box_path_list = glob.glob(img_dir + f"/coco_*.jpg")[::3]
        self.transform = transform

    def __len__(self):
        return len(self.box_path_list)

    def __getitem__(self, idx):
        image = read_image(self.box_path_list[idx]) / 255.0
        image_cut = cut_out(image)
        if self.transform:
            image = self.transform(image)
            image_cut = self.transform(image_cut)
        return image_cut, image


def get_file_index(x):
    """Get file index from file name

    Args:
        x (str): _description_

    Returns:
        tuple: tuple containing
            int: file index1
            int: file index2
    """
    idx1 = int(os.path.basename(x).split("_")[0])
    idx2 = int(os.path.basename(x).split("_")[1].split(".")[0])
    return idx1, idx2


class LabelConverterAtt:
    """Convert label into Attention Format"""

    def __init__(self):
        self.max_len = 15
        character = list(string.printable[:-5])
        list_token = ["[GO]", "[s]"]
        self.character = list_token + character
        self.dictionary = {}

        for i, char in enumerate(self.character):
            self.dictionary[char] = i

    def encode(self, text):
        encoded = torch.LongTensor(1, self.max_len + 2).fill_(0)  # N: 1(batch)
        text = list(text)
        text.append("[s]")
        text = [self.dictionary[char] for char in text]
        encoded[0][1 : 1 + len(text)] = torch.LongTensor(text)
        return encoded[0]

    def decode(self, t):
        text = "".join([self.character[i] for i in t if i > 1])
        return text


class TigerDataset(Dataset):
    """Dataset for Tiger

    Tiger Dataset with Image and text
    """

    def __init__(
        self,
        img_path="dataset/tiger/images",
        gt_path="dataset/tiger/gt.txt",
        transform=None,
        encode=True,
    ):
        with open(gt_path) as f:
            self.gt_tiger = f.readlines()
        self.gt_tiger = [x.rstrip().split("\t") for x in self.gt_tiger]
        self.img_path_lst = sorted(glob.glob(img_path + "/*/*"), key=get_file_index)
        self.transform = transform
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((32, 128)),
                transforms.ToTensor(),
            ]
        )
        self.converter = LabelConverterAtt()
        self.encode = encode

    def __len__(self):
        return len(self.gt_tiger) // 2

    def __getitem__(self, idx):
        index = idx * 2

        # windows path problem
        image1 = read_image(self.img_path_lst[index].replace("\\", "/"))
        image2 = read_image(self.img_path_lst[index + 1].replace("\\", "/"))
        # image1 = read_image(self.img_path_lst[index].replace(os.sep, "/"))
        # image2 = read_image(self.img_path_lst[index + 1].replace(os.sep, "/"))

        image1 = self.preprocess(image1)
        image2 = self.preprocess(image2)

        label1 = self.gt_tiger[index][1]
        label2 = self.gt_tiger[index + 1][1]
        length1 = len(label1)

        if self.encode:
            label1 = self.converter.encode(label1)
            length1 += 1

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, label1, image2, label2, length1


def get_synth_dataset():
    mean = [0.5505, 0.5470, 0.5305]
    std = [0.2902, 0.3023, 0.2819]

    synth_dataset = TigerDataset(
        img_path="Synth/synthtiger/results/images",
        gt_path="Synth/synthtiger/results/gt.txt",
        transform=transforms.Normalize(mean, std),
    )
    return synth_dataset


def get_real_dataset():
    mean = [0.5041, 0.4809, 0.4665]
    std = [0.2729, 0.2632, 0.2728]

    ic13_dataset = ICDataset(
        "dataset/cropped", tag="IC13", transform=transforms.Normalize(mean, std)
    )
    ic15_dataset = ICDataset(
        "dataset/cropped", tag="IC15", transform=transforms.Normalize(mean, std)
    )
    ic17_dataset = ICDataset(
        "dataset/cropped", tag="IC17", transform=transforms.Normalize(mean, std)
    )
    ic19_dataset = ICDataset(
        "dataset/cropped", tag="IC19", transform=transforms.Normalize(mean, std)
    )
    coco_dataset = COCODataset(
        img_dir="dataset/cropped", transform=transforms.Normalize(mean, std)
    )
    real_dataset = torch.utils.data.ConcatDataset(
        [ic13_dataset, ic15_dataset, ic17_dataset, ic19_dataset, coco_dataset]
    )
    return real_dataset


def get_real_testset():
    mean = [0.5041, 0.4809, 0.4665]
    std = [0.2729, 0.2632, 0.2728]
    ic13_dataset = ICDataset(
        "dataset/cropped_test", tag="IC13", transform=transforms.Normalize(mean, std)
    )
    ic15_dataset = ICDataset(
        "dataset/cropped_test", tag="IC15", transform=transforms.Normalize(mean, std)
    )
    return ic13_dataset, ic15_dataset
