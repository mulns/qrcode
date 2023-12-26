from collections import OrderedDict
from pprint import pprint

import torch
import torch.nn.functional as F
from torchvision import transforms

import utils


def transform(x, module_num=37, module_size=16, margin=0):
    # x: PIL Image in shape (H W C)
    x = transforms.ToTensor()(x)  # (C H W)
    new_size = (module_num + margin * 2) * module_size
    resized = F.interpolate(
        x[None], size=(new_size, new_size), mode="bilinear", align_corners=True
    )
    return resized.to(DEVICE)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MN = 37
MS = 16
MG = 5
tools = utils.QRTools(MS, 6, MG)


def calc_error(Q, C):
    # Q = "./out/final/results/01543.png"
    # C = "./out/final/blueprint/01543.png"
    Q = transform(utils.load_image(Q), MN, MS, MG).to(DEVICE)
    C = transform(utils.load_image(C), MN, MS, MG).to(DEVICE)
    Q_y = tools.convert2L(Q)
    Q_b = torch.zeros_like(Q_y, dtype=Q.dtype, device=Q.device)
    Q_b[Q_y >= 0.5] = 1.0

    center = tools.get_center_pixel(Q_y)
    boundary = [0.0, tools._norm(100), tools._norm(150), 1.0]
    boundary = torch.tensor(boundary, device=center.device, dtype=center.dtype)
    MQ = torch.bucketize(center, boundary) - 2

    MC = tools.get_binary_result(C) * 2 - 1
    MQ = MQ[..., MG:-MG, MG:-MG]
    MC = MC[..., MG:-MG, MG:-MG]
    att = 1.0 - (MQ == MC).float()
    return att.sum()


def main():
    out_dict = OrderedDict()
    ids = "01425,01433,01461,01470,01554,01643,01653,01705,01717,01688"
    # ids = "01705,01717,01688"
    for id in ids.split(","):
        out_dict[id] = OrderedDict()
        for type in ["2", "5", "8", "11", "14"]:
            Q = f"data/paper/{id}/pic{type}.png"
            C = f"data/paper/{id}/qr{type}_margin5.png"
            e = calc_error(Q, C).item() / (37**2)
            out_dict[id][type] = f"{e:.3f}"

    pprint(out_dict)


if __name__ == "__main__":
    main()
