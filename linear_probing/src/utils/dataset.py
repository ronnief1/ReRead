import os
import random
import torchvision.transforms.functional as F

import torch
from torch import nn, Tensor
from torchvision import datasets
from torchvision.datasets.folder import default_loader
import torchvision.transforms as tvtr
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



class PartialImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        percentage=0.5,
    ):
        super(PartialImageFolder, self).__init__(
            root, transform, target_transform, loader
        )
        self.percentage = percentage
        self.sample_indices = self._generate_sample_indices()

    def _generate_sample_indices(self):
        sample_indices = {}
        for target_class in self.classes:
            class_dir = os.path.join(self.root, target_class)
            all_images = os.listdir(class_dir)
            num_samples = int(len(all_images) * self.percentage)
            sample_indices[target_class] = random.sample(
                range(len(all_images)), num_samples
            )
        return sample_indices

    def __getitem__(self, index):
        print(self.classes)
        path, target = self.samples[index]
        class_samples = self.sample_indices[
            self.classes[target]
        ]  # Get samples for the target class
        sample_index = class_samples[
            index % len(class_samples)
        ]  # Use modulo to handle index out of range
        sample_path = os.path.join(self.samples[sample_index][0])
        sample = self.loader(sample_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def build_dataset(subset, args, augment=False):  #, visualisation=False):
    # transform = (
    #     build_transform(args, is_train) if not visualisation else build_visual_transform(args)
    # )
    transform = build_transform(args, subset, augment)
    root = os.path.join(args.data_path, subset)
    # print(root)
    if subset == "train" and args.label_efficiency_exp:
        dataset = PartialImageFolder(
            root, transform=transform, percentage=args.train_ds_perc
        )
    else:
        dataset = datasets.ImageFolder(root, transform=transform)
    # print(len(dataset))
    return dataset


class MinMaxScaler:
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())


class MinMaxScalerChannel:
    def __init__(self) -> None:
        super().__init__()
        self.scaler = MinMaxScaler()

    def __call__(self, tensor):
        for i in range(tensor.shape[0]):
            if tensor[i].max() > 0:
                tensor[i] = self.scaler(tensor[i:i+1].clone())
        return tensor


class NaiveScaler:
    """
    Transforms each channel to the range [0, 1], if it is not already.
    """
    def __call__(self, tensor):
        if tensor.min() < 0:
            raise ValueError("Tensor contains negative values")
        elif tensor.max() > 1 and tensor.max() <= 255:
            tensor = tensor / 255.0
        elif tensor.max() > 255:
            tensor = tensor / 65535.0
        return tensor


class NaiveScalerChannel:
    """
    Transforms each channel to the range [0, 1], if it is not already.
    """
    def __init__(self) -> None:
        super().__init__()
        self.scaler = NaiveScaler()

    def __call__(self, tensor):
        for i in range(tensor.shape[0]):
            tensor[i] = self.scaler(tensor[i:i+1].clone())
        return tensor


class Identity:
    def __call__(self, img):
        return img


class ToRGB:
    def __call__(self, img: torch.Tensor):
        return img.repeat(3, 1, 1)


class RandomIntensity(nn.Module):
    def __init__(self, intensity_range=(0.8, 1.2)):
        super().__init__()
        self.intensity_range = intensity_range

    @staticmethod
    def get_abs_max(tensor):
        if tensor.max() <= 1:
            abs_max = 1
        elif tensor.max() > 1 and tensor.max() <= 255:
            abs_max = 255
        elif tensor.max() > 255:
            abs_max = 65535
        else:
            raise ValueError(
                "Image values are not in the expected range:"
                f" [{tensor.max()}, {tensor.min()}], {torch.unique(tensor)}"
            )
        return abs_max

    def forward(self, img):
        intensity = torch.empty(1).uniform_(*self.intensity_range).item()
        return torch.clamp(img * intensity, 0, self.get_abs_max(img))


class RandomIntensityChannel(nn.Module):
    def __init__(self, intensity_range=(0.8, 1.2)):
        super().__init__()
        self.intensity_range = intensity_range
        self.intensity = RandomIntensity(intensity_range)

    def forward(self, img):
        for i in range(img.shape[0]):
            if img[i].max() > 0:
                img[i] = self.intensity(img[i:i+1].clone())
        return img


class RandomAffineChannel(tvtr.RandomAffine):
    """Same as RandomAffine but with a random rotation for every
    channel.
    """
    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)]
            else:
                fill = [float(f) for f in fill]  # type: ignore

        img_size = F.get_image_size(img)

        for i in range(img.shape[0]):
            # Apply a transformation only in 90% of the cases
            if random.random() < 0.9:
                ret = self.get_params(
                    self.degrees, self.translate, self.scale, self.shear,
                    img_size
                )
                img[i] = F.affine(
                    img[i:i+1].clone(), *ret, interpolation=self.interpolation,
                    fill=fill, center=self.center  # type: ignore
                )
        return img


def build_transform(args, subset, augment):
    multimodal = len(args.input_modality.split("-")) > 1
    if multimodal:
        end = 'for multimodal data'
    else:
        end = ''
    print(f">>> Building transform '{subset}'", end)
    intensity_msg = 'Random intensity shift'
    intensity = RandomIntensityChannel()
    if args.fill is None:
        if 'kermany' in args.data_set.lower() or '_shuffle_' in args.init_weights:
            fill = 1
        else:
            fill = 0
    else:
        fill = args.fill
    affine_msg = f'Random affine (fill={fill})'
    affine = RandomAffineChannel(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=5,
        interpolation=tvtr.InterpolationMode.BILINEAR,
        fill=fill
    )
    if args.no_affine:
        affine_msg = 'No random affine'
        affine = Identity()
    grayscale = Identity()
    min_max = Identity()
    scaler_list = [ NaiveScalerChannel() ]
    if not multimodal:
        grayscale = tvtr.Grayscale(num_output_channels=1)
    # If it is a SOTA model
    scaler_msg = "ImageNet scaler"
    if not multimodal:
        # First to RGB, then normalize
        scaler_list += [ ToRGB() ]

    if 'vfm' in args.init_weights.lower():
        print('Using visionFM norm mean and sd')
        norm_mean = (0.21091926, 0.21091926, 0.21091919)
        norm_sd = (0.17598894, 0.17598891, 0.17598893)
    elif 'clip' in args.init_weights.lower():
        print('Using CLIP norm mean and sd')
        norm_mean =  (0.48145466, 0.4578275, 0.40821073)
        norm_sd =  (0.26862954, 0.26130258, 0.27577711)
    else:
        print('Using Imagenet norm mean and sd')
        norm_mean = IMAGENET_DEFAULT_MEAN
        norm_sd = IMAGENET_DEFAULT_STD
    scaler_list += [
        tvtr.Normalize(norm_mean, norm_sd)
    ]

    transforms_list = [
        tvtr.Resize(
            size=(args.input_size, args.input_size),
            interpolation=tvtr.InterpolationMode.BILINEAR,
        ),
        grayscale, # what is this
        tvtr.ToTensor(),
        tvtr.ConvertImageDtype(torch.float32),
        min_max,
    ]
    if augment:
        print("Random horizontal flip (0.5)")
        print(intensity_msg)
        print(affine_msg)
        transforms_list += [
            tvtr.RandomHorizontalFlip(p=0.5),
            intensity,
            affine,
        ]
    print(scaler_msg)
    transforms_list += scaler_list
    transforms = tvtr.Compose(transforms_list)

    return transforms

