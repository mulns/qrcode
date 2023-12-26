import argparse
import os
import pprint
import shutil

import diffusers
import lpips
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import utils
from loss import Loss


def transform(x, module_num=37, module_size=16, margin=0):
    # x: PIL Image in shape (H W C)
    x = transforms.ToTensor()(x)  # (C H W)
    new_size = (module_num + margin * 2) * module_size
    resized = F.interpolate(
        x[None], size=(new_size, new_size), mode="bilinear", align_corners=True
    )
    return resized.to(DEVICE)


class Updater(object):
    def __init__(self, content_pth, code_pth, target_pth, out_dir, configs):
        self.vae_config = configs.vae_cfg.params
        self.args = configs.artcode_cfg.params

        self.out_dir = out_dir
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        self._build_assets()
        target_pth = content_pth if target_pth is None else target_pth
        self._initialize(content_pth, code_pth, target_pth)
        self._to(DEVICE)

    def _build_assets(self):
        if self.args.use_vae:
            # url = "checkpoints/vae-ft-mse-840000-ema-pruned.safetensors"
            url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
            # can also be a local file

            self.vae = diffusers.models.AutoencoderKL.from_single_file(url)

            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.to(DEVICE).eval()
        else:
            self.vae = None

        # the qrcode tool
        self.qrtool = utils.QRTools(
            module_size=MODULE_SIZE, center_size=CENTER_SIZE, margin=MARGIN
        )

        # loss functions
        self.cd_loss = Loss(self.args.code_loss, module_size=MODULE_SIZE, margin=MARGIN)
        self.ct_loss = Loss(self.args.content_loss)
        # initialize the visualizer
        self.visualizer = utils.VisualizeLogger(self.out_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(self.out_dir, "tb"))
        return self

    def _initialize(self, img_pth, code_pth, target_pth):
        img = utils.load_image(filename=img_pth)
        code = utils.load_image(filename=code_pth)
        target = utils.load_image(filename=target_pth)
        self.org_size = img.size

        img = transform(img, MODULE_NUM, MODULE_SIZE, margin=MARGIN)
        code = transform(code, MODULE_NUM, MODULE_SIZE, margin=MARGIN)
        target = transform(target, MODULE_NUM, MODULE_SIZE, margin=MARGIN)
        self.code = code

        # initialize code and content target
        self.code_target = self.qrtool.get_target(
            code, self.args.correct_b, self.args.correct_w
        )

        # initialize latent code to be updated
        mask = utils.get_pattern_center_mask(img, MARGIN, MODULE_SIZE, MODULE_SIZE // 8)
        # mask = utils.get_pattern_mask(img, MARGIN, MODULE_SIZE, MODULE_SIZE // 8)
        img = mask * self.code_target + (1 - mask) * img
        lc: torch.Tensor = self._encode(img).detach()
        self.lc = lc.requires_grad_()
        self.out = torch.clamp(self._decode(self.lc).data, 0.0, 1.0)

        utils.save_tensor(
            self.code_target, self.out_dir, "code_target.png", size=self.org_size
        )

        self.img_target = target

        # initialize the optimizer
        self.optimizer = optim.Adam([self.lc], lr=self.args.learning_rate)

    def _to(self, device):
        self.cd_loss.to(device)
        self.ct_loss.to(device)
        self.qrtool.to(device)
        # if self.vae is not None:
        #     self.vae.to(device)
        return self

    def _encode(self, x):
        if self.args.use_vae:
            with torch.no_grad():
                posterior = self.vae.encode(x).latent_dist
                x = posterior.mode()
            return x
        else:
            return x

    def _decode(self, x):
        if self.args.use_vae:
            return self.vae.decode(x).sample
        else:
            return x

    def iter(self, step):
        decoded = self._decode(self.lc)
        self.out = torch.clamp(decoded.data, 0.0, 1.0)

        content_loss, _ = self.ct_loss(decoded, self.img_target)

        att = self.qrtool.get_error_module(
            decoded, self.code_target, self.args.threshold_b, self.args.threshold_w
        )

        code_loss, _ = self.cd_loss(decoded, self.code_target, att)

        if MARGIN > 0:
            att = att[..., MARGIN:-MARGIN, MARGIN:-MARGIN]

        psnr = utils.calc_psnr(self.out, self.img_target)
        with torch.no_grad():
            lpips = calc_lpips(self.out, self.img_target).item()
        self.writer.add_scalar("CTLoss", content_loss, step)
        self.writer.add_scalar("CDLoss", code_loss, step)
        self.writer.add_scalar("PSNR", psnr, step)
        self.writer.add_scalar("LPIPS", lpips, step)
        self.writer.add_scalar("Errors", att.sum(), step)
        return dict(
            code_loss=code_loss,
            content_loss=content_loss,
            att=att,
            psnr=psnr,
            lpips=lpips,
        )

    def run(self, steps):
        print("START".center(100, "="))
        for epoch in range(steps + 1):
            self.optimizer.zero_grad()
            metrics = self.iter(epoch)
            content_loss = metrics["content_loss"]
            code_loss = metrics["code_loss"]
            att = metrics["att"]
            total_loss = (
                self.args.content_weight * content_loss
                + self.args.code_weight * code_loss
            )
            total_loss.backward(retain_graph=False)
            self.optimizer.step()

            ################### Visualize ###################
            if epoch % 10 == 0:
                print(
                    "Iter {}: Content Loss: {:.4f}. PSNR/LPIPS: {:.2f}/{:.4f}. Code Loss: {:.4f}. Error module number: {:4.0f} / {:4.0f}.".format(
                        epoch,
                        content_loss,
                        metrics["psnr"],
                        metrics["lpips"],
                        code_loss,
                        att.sum(),
                        att.cpu().numpy().size,
                    )
                )
            if epoch % 50 == 0:
                feas = dict(
                    latent_code=self.lc.detach(), gradients=self.lc.grad.detach()
                )
                self.visualizer.update(feas, epoch)
            if epoch % 20 == 0:
                # if epoch % 100 == 0:
                img_name = "it%.3d-e%.4d.jpg" % (epoch, att.sum())
                utils.save_tensor(self.out, self.out_dir, img_name, self.org_size)
                # img_name = "it%.3d-e%.4d-pt.jpg" % (epoch, att.sum())
                # out = self.qrtool.add_pattern(self.out, self.code_target, padding=2)
                # utils.save_tensor(out, self.out_dir, img_name, self.org_size)
                print("Save output: " + img_name + " In dir: " + self.out_dir)

        ##### visualize the changing of latent code and its gradients #####
        self.visualizer.imshow()
        self.writer.flush()
        self.writer.close()


def main(args):
    with open(args.config, "r") as f:
        configs = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    pprint.pprint(configs)

    updater = Updater(
        content_pth=args.content,
        code_pth=args.code,
        target_pth=args.target,
        out_dir=args.dir,
        configs=configs,
    )
    updater.run(args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--content", "-ct", type=str, default="data/datasets/QR1200/samples/82/82_2.png"
    )
    parser.add_argument(
        "--code", "-cd", type=str, default="data/datasets/QR1200/samples/82/82_1.png"
    )
    parser.add_argument("--target", "-tg", type=str, default=None)

    parser.add_argument("--dir", "-dir", type=str, default="./82")
    parser.add_argument("--epochs", "-e", type=int, default=100)

    parser.add_argument("--margin", "-mg", type=int, default=0)
    parser.add_argument("--module_size", "-ms", type=int, default=16)
    parser.add_argument("--center_size", "-cs", type=int, default=6)
    parser.add_argument("--module_num", "-mn", type=int, default=37)
    args = parser.parse_args()

    MODULE_NUM = args.module_num  # Defined by the QRCode version. Version 5 -> 37
    MARGIN = args.margin  # Generated QR Code includes margins.
    MODULE_SIZE = args.module_size  # Customized pixel size per module.
    CENTER_SIZE = args.center_size  # Valid area of an module
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calc_lpips = lpips.LPIPS(net="alex").to(DEVICE)
    main(args)
