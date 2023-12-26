import math
import os

import numpy as np
import torch
from easydict import EasyDict
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid


def calc_psnr(a, b):
    a = torch.clamp(a, 0.0, 1.0) * 255.0
    b = torch.clamp(b, 0.0, 1.0) * 255.0
    mse = torch.mean((a - b) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize(
            (int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS
        )
    return img


def get_pattern_mask(x, margin, module_size, padding=0):
    assert padding < module_size, "padding cannot greater than module_size"
    # x: NxCxHxW / CxHxW
    # Only works when QRCode is version 5

    full_marker = torch.zeros_like(x, dtype=x.dtype, device=x.device)

    # Locating Marker
    if margin == 0:
        x1 = 0
        x2 = 7 * module_size + padding
        full_marker[..., :x2, :x2] = 1
        full_marker[..., :x2, -x2:] = 1
        full_marker[..., -x2:, :x2] = 1
    else:
        x1 = margin * module_size - padding
        x2 = margin * module_size + 7 * module_size + padding
        full_marker[..., x1:x2, x1:x2] = 1
        full_marker[..., x1:x2, -x2:-x1] = 1
        full_marker[..., -x2:-x1, x1:x2] = 1

    # Alignment Marker
    x1 = margin * module_size + 28 * module_size - padding
    x2 = margin * module_size + 33 * module_size + padding
    full_marker[..., x1:x2, x1:x2] = 1
    return full_marker


def get_pattern_center_mask(x, margin, module_size, padding=0):
    # x: NxCxHxW / CxHxW
    # Only works when QRCode is version 5

    center_marker = torch.zeros_like(x, dtype=x.dtype, device=x.device)

    # Locating Marker
    n, c = x.shape[:2]
    cross_stitch = torch.zeros(
        (n, c, 2 * padding + 7 * module_size, 2 * padding + 7 * module_size),
        dtype=x.dtype,
        device=x.device,
    )
    cross_stitch[..., padding + 3 * module_size : padding + 4 * module_size] = 1
    cross_stitch[..., padding + 3 * module_size : padding + 4 * module_size, :] = 1
    if margin == 0:
        x1 = 0
        x2 = 7 * module_size + padding
        if padding == 0:
            center_marker[..., :x2, :x2] = cross_stitch
            center_marker[..., :x2, -x2:] = cross_stitch
            center_marker[..., -x2:, :x2] = cross_stitch
        else:
            center_marker[..., :x2, :x2] = cross_stitch[..., padding:, padding:]
            center_marker[..., :x2, -x2:] = cross_stitch[..., :-padding, padding:]
            center_marker[..., -x2:, :x2] = cross_stitch[..., padding:, :-padding]
    else:
        x1 = margin * module_size - padding
        x2 = margin * module_size + 7 * module_size + padding
        center_marker[..., x1:x2, x1:x2] = cross_stitch
        center_marker[..., x1:x2, -x2:-x1] = cross_stitch
        center_marker[..., -x2:-x1, x1:x2] = cross_stitch

    # Alignment Marker
    cross_stitch = torch.zeros(
        (n, c, 5 * module_size + 2 * padding, 5 * module_size + 2 * padding),
        dtype=x.dtype,
        device=x.device,
    )
    cross_stitch[..., padding + 2 * module_size : padding + 3 * module_size] = 1
    cross_stitch[..., padding + 2 * module_size : padding + 3 * module_size, :] = 1
    x1 = margin * module_size + 28 * module_size - padding
    x2 = margin * module_size + 33 * module_size + padding
    center_marker[..., x1:x2, x1:x2] = cross_stitch
    return center_marker


def save_tensor(x, path, name, size=None):
    image = x[0].cpu().clone()
    image = transforms.ToPILImage()(image)
    if size is not None:
        image = image.resize(size, Image.LANCZOS)
    image.save(os.path.join(path, str(name)))


class MarginedPool2D(torch.nn.Conv2d):
    def __init__(self, in_channels, kernel_size, margin, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
            dilation=1,
            groups=in_channels,
            bias=False,
            padding_mode="zeros",
            **kwargs,
        )
        self.weight = torch.nn.Parameter(
            self.get_kernel(in_channels, self.kernel_size, margin),
            requires_grad=False,
        )

    def get_kernel(self, in_channels, kernel_size, margin):
        if not isinstance(margin, (tuple, list)):
            margin = (margin, margin)
        ksize = (kernel_size[0] - 2 * margin[0], kernel_size[1] - 2 * margin[1])
        kernel = torch.ones((in_channels, 1, *ksize))
        kernel = torch.nn.functional.pad(
            kernel,
            pad=(margin[1], margin[1], margin[0], margin[0], 0, 0, 0, 0),
            mode="constant",
            value=0,
        )
        w = ksize[0] * ksize[1]
        return kernel / w


class GaussianPool2D(torch.nn.Conv2d):
    def __init__(self, in_channels=3, module_size=8, sigma=1.5):
        super().__init__(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=False,
        )

        weight = self._get_3DGauss(module_size, sigma=sigma)
        self.weight = torch.nn.Parameter(
            weight.repeat(1, in_channels, 1, 1), requires_grad=False
        )

    def _get_3DGauss(self, size=8, sigma=1.5):
        s = 0
        e = size - 1
        mu = e / 2
        x, y = np.mgrid[s : e : size * (1j), s : e : size * (1j)]
        z = (1 / (2 * math.pi * sigma**2)) * np.exp(
            -((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma**2)
        )
        z = torch.from_numpy(self._norm(z.astype(np.float32)))
        for j in range(size):
            for i in range(size):
                if z[i, j] < 0.1:
                    z[i, j] = 0
        return z

    def _norm(self, x):
        maxvalue = np.max(x)
        minvalue = np.min(x)
        x = (x - minvalue) / (maxvalue - minvalue)
        x = np.around(x, decimals=2)
        return x


class QRTools(object):
    def __init__(self, module_size, center_size=6, margin=0):
        super().__init__()
        self.module_size = module_size
        _margin = (module_size - center_size) // 2
        self.extract = MarginedPool2D(1, module_size, _margin)
        self.margin = margin

    def _norm(self, x):
        if x > 1:
            return x / 255.0
        else:
            return x

    def convert2L(self, x):
        # x: N 3 H W (rgb)
        if x.shape[1] == 3:
            xs = x.chunk(3, dim=1)
            return 0.299 * xs[0] + 0.587 * xs[1] + 0.114 * xs[2]
        elif x.shape[1] == 1:
            return x

    def get_center_pixel(self, x):
        # x: N1HW
        x = self.convert2L(x)
        self.extract.to(x.device)
        return self.extract(x)

    def get_binary_result(self, x):
        # x: N1HW
        x = self.convert2L(x)
        center = self.get_center_pixel(x)
        out = torch.zeros_like(center, dtype=center.dtype, device=center.device)
        out[center >= 0.5] = 1.0
        return out.float()

    def get_error_module(self, img, code, threshold_b, threshold_w):
        binary = self.get_binary_result(code)
        sb_code = 2 * binary - 1.0

        center = self.get_center_pixel(img)
        boundary = [0.0, self._norm(threshold_b), self._norm(threshold_w), 1.0]
        boundary = torch.tensor(boundary, device=center.device, dtype=center.dtype)
        sb_center = (torch.bucketize(center, boundary) - 2).float()

        error_module = 1.0 - (sb_center == sb_code).float()
        marker_mask = get_pattern_mask(error_module, self.margin, 1, 0)
        return error_module * (1 - marker_mask)

    def get_target(self, code, b_robust, w_robust):
        margin = self.margin
        binary = self.get_binary_result(code)
        if not margin == 0:
            target = torch.ones_like(binary, device=binary.device, dtype=binary.dtype)
            target[..., margin:-margin, margin:-margin] = binary[
                ..., margin:-margin, margin:-margin
            ]
        else:
            target = binary
        up = torch.nn.functional.interpolate(
            target, scale_factor=self.module_size, mode="nearest"
        )
        return torch.clamp(up, self._norm(b_robust), self._norm(w_robust)).repeat(
            1, 3, 1, 1
        )

    def add_pattern(self, target, code, padding):
        mask = get_pattern_center_mask(code, self.margin, self.module_size, padding)
        out = mask * code + (1 - mask) * target
        return out

    def to(self, device):
        self.extract.to(device)
        return self


class VisualizeLogger(object):
    def __init__(self, save_dir="", epsilon=1e-4):
        self.data = EasyDict(
            dict(latent_code=[], latent_diff=[], gradients=[], gradients_sign=[])
        )
        # latent_code: tensors in 1x4xHxW
        # gradients: tensors in 1x4xHxW
        # steps: recording steps
        self.steps = []
        self.save_dir = save_dir
        self.epsilon = epsilon

    def _sign(self, x, epsilon=None):
        epsilon = self.epsilon if epsilon is None else epsilon
        boundaries = [-1e10, -epsilon, epsilon, 1e10]
        boundaries = torch.tensor(boundaries, device=x.device, dtype=x.dtype)
        sign = (torch.bucketize(x, boundaries) - 2).float()
        return sign

    def update(self, x: dict, step: int):
        self.data.latent_code.append(x["latent_code"].cpu())
        if len(self.data.latent_code) == 1:
            compared = self.data.latent_code[-1]
        else:
            compared = self.data.latent_code[-2]
        self.data.latent_diff.append(compared - x["latent_code"].cpu())
        self.data.gradients.append(x["gradients"].cpu())
        self.data.gradients_sign.append(self._sign(x["gradients"].cpu()))
        self.steps.append(step)

    def imshow(self):
        for name in self.data.keys():
            tensor_list = []
            for tensor in self.data[name]:
                tensor = tensor.permute(1, 0, 2, 3)  # 4 x 1 x H x W
                flatten = make_grid(tensor, nrow=2, normalize=False, padding=0)
                # 1 x 1 x 2H x 2W
                tensor_list.append(flatten)
            all_tensor = make_grid(tensor_list, nrow=len(self), normalize=True)
            transforms.ToPILImage()(all_tensor).save(
                os.path.join(self.save_dir, "%s.png" % name)
            )

    def __len__(self):
        return len(self.data)

    def __iter__(self, index):
        out = {}
        for k, v in self.data.items():
            out[k] = v[index]
        return out
