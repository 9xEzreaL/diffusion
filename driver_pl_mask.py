import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.optim import AdamW
from lion_pytorch import Lion
# from med_seg_diff_pytorch import Unet, MedSegDiff
from med_seg_diff_pytorch.diff_mask import MedSegDiff, Unet
import cv2
import pytorch_lightning as pl
import random

## Parse CLI arguments ##
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-slr', '--scale_lr', action='store_true', help="Whether to scale lr.")
    parser.add_argument('-rt', '--report_to', type=str, default="wandb", choices=["wandb"],
                        help="Where to log to. Currently only supports wandb")
    parser.add_argument('-ld', '--logging_dir', type=str, default="/media/ziyi/logs/diffusion-logs", help="Logging dir.")
    parser.add_argument('-od', '--output_dir', type=str, default="output", help="Output dir.")
    parser.add_argument('-mp', '--mixed_precision', type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to do mixed precision")
    parser.add_argument('-ga', '--gradient_accumulation_steps', type=int, default=4,
                        help="The number of gradient accumulation steps.")
    parser.add_argument('-img', '--img_folder', type=str, default='/media/ziyi/Dataset/TSE_DESS/TSE_DESS/train',
                        help='The image file path from data_path')
    parser.add_argument('-csv', '--csv_file', type=str, default='ISBI2016_ISIC_Part3B_Training_GroundTruth.csv',
                        help='The csv file to load in from data_path')
    parser.add_argument('-sc', '--self_condition', action='store_true', help='Whether to do self condition')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.95,
                        help='The beta1 parameter for the Adam optimizer.')
    parser.add_argument('-ab2', '--adam_beta2', type=float, default=0.999,
                        help='The beta2 parameter for the Adam optimizer.')
    parser.add_argument('-aw', '--adam_weight_decay', type=float, default=1e-6,
                        help='Weight decay magnitude for the Adam optimizer.')
    parser.add_argument('-ae', '--adam_epsilon', type=float, default=1e-08,
                        help='Epsilon value for the Adam optimizer.')
    parser.add_argument('-ul', '--use_lion', type=bool, default=False, help='use Lion optimizer')
    parser.add_argument('-ic', '--mask_channels', type=int, default=1, help='input channels for training (default: 3)')
    parser.add_argument('-c', '--input_img_channels', type=int, default=1,
                        help='output channels for training (default: 3)')
    parser.add_argument('-is', '--image_size', type=int, default=256, help='input image size (default: 128)')
    parser.add_argument('-dd', '--data_path', default='/media/ziyi/Dataset/TSE_DESS/TSE_DESS/train',
                        help='directory of input image')
    parser.add_argument('-d', '--dim', type=int, default=64, help='dim (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs (default: 10000)')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='batch size to train on (default: 8)')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps (default: 1000)')
    parser.add_argument('-ds', '--dataset', default='oai', help='Dataset to use')
    parser.add_argument('--save_every', type=int, default=10, help='save_every n epochs (default: 100)')
    parser.add_argument('--load_model_from', default=None, help='path to pt file to load from')
    parser.add_argument('-exp', '--exp', default='oai', help='Experiment name')
    return parser.parse_args()


def load_data(args):
    # Load dataset
    if args.dataset == 'ISIC':
        transform_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), ]
        transform_train = transforms.Compose(transform_list)
        dataset = ISICDataset(args.data_path, args.csv_file, args.img_folder, transform=transform_train, training=True,
                              flip_p=0.5)
    elif args.dataset == 'generic':
        transform_list = [transforms.ToPILImage(), transforms.Resize(args.image_size), transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        dataset = GenericNpyDataset(args.data_path, transform=transform_train, test_flag=False)
    elif args.dataset == 'oai':
        from med_seg_diff_pytorch.dataset import OAIDataset_womask
        dataset = OAIDataset_womask(args.data_path, 'train')
    elif args.dataset == 'pain':
        from med_seg_diff_pytorch.dataset import OAI_pain
        dataset = OAI_pain(args.data_path, 'train')
        test_dataset = OAI_pain(args.data_path, 'test')
        test_generator = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False)
    else:
        raise NotImplementedError(f"Your dataset {args.dataset} hasn't been implemented yet.")

    ## Define PyTorch data generator
    training_generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True)
    if args.dataset == 'pain':
        return training_generator, test_generator
    else:
        return training_generator


def to_8bit(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return img

class DiffusionPl(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        ## Config model ##
        self.model = model
        self.checkpoint_dir = os.path.join(args.output_dir, args.exp, 'checkpoints')
        self.logging_dir = os.path.join(args.output_dir, args.exp, args.logging_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.diffusion = MedSegDiff(
            self.model,
            timesteps=args.timesteps
        ).cuda()
        self.epoch = 0
        self.running_loss = 0.0
        self.total_count = 0


    def configure_optimizers(self):
        ## Initialize optimizer
        if not self.args.use_lion:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.adam_weight_decay,
                eps=self.args.adam_epsilon,
            )
        else:
            optimizer = Lion(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.adam_weight_decay
            )
        return optimizer

    def training_step(self, batch, batch_idx):
        dess_img, tse_img = batch
        mask = ''
        dess_img = dess_img.float()
        tse_img = tse_img.float()

        loss = self.diffusion(dess_img, mask, tse_img)
        self.log('train-loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        # For visualize
        self.running_loss += loss.item() * dess_img.size(0)
        self.total_count += tse_img.shape[0]
        return loss

    def training_epoch_end(self, x):
        epoch_loss = self.running_loss / self.total_count
        self.log('avg-train-loss', epoch_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.total_count = 0
        self.running_loss = 0
        self.epoch += 1
        print('---------------------------')
        if self.epoch % self.args.save_every == 0:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.diffusion.model.state_dict(),
                # 'optimizer_state_dict': self.optimizer.state_dict(),
                # 'loss': self.loss,
            }, os.path.join(self.checkpoint_dir, f'state_dict_epoch_{self.epoch}_loss_{epoch_loss}.pt'))

    def validation_step(self, batch, batch_idx=0):
        num = random.randint(0, 20)
        dess_img, tse_img, _ = batch
        dess_img = dess_img.float()
        tse_img = tse_img.float()
        pred_noise, t2d_img = self.diffusion.sample(tse_img)
        pred_noise = pred_noise.cpu().detach().numpy()
        t2d_img = t2d_img.cpu().detach().numpy()

        visualize_list = []
        for i in range(t2d_img.shape[0]):
            tmp = np.concatenate([to_8bit(tse_img[i, ::].cpu().detach().numpy() - t2d_img[i, ::]), to_8bit(dess_img[i:i + 1, ::].cpu().detach().numpy()),
                                  to_8bit(tse_img[i:i + 1, ::].cpu().detach().numpy())], 1)
            visualize_list.append(tmp)
        tmp = np.concatenate(visualize_list, 2)
        cv2.imwrite(f'output/{self.args.exp}/{self.epoch}_{num}.png', np.transpose(tmp, (1, 2, 0)))


if __name__ == '__main__':
    args = parse_args()
    checkpoint_dir = os.path.join(args.output_dir, args.exp, 'checkpoints')
    logging_dir = os.path.join(args.output_dir, args.exp, args.logging_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    ## DEFINE MODEL ##
    model = Unet(
        dim=args.dim,
        image_size=args.image_size,
        dim_mults=(1, 2, 4, 8),
        mask_channels=args.mask_channels,
        input_img_channels=args.input_img_channels,
        self_condition=args.self_condition
    )

    ## LOAD DATA ##
    if args.dataset == 'pain':
        data_loader, test_loader = load_data(args)
        test_dess_img, test_tse_img, id = next(iter(test_loader))
    else:
        data_loader = load_data(args)

    ## TRAIN MODEL ##
    counter = 0
    diffusion = MedSegDiff(
        model,
        timesteps=args.timesteps
    )

    diffusion_pl = DiffusionPl(args, model)
    trainer = pl.Trainer(gpus=1, accelerator='ddp',
                         max_epochs=500, progress_bar_refresh_rate=4, log_every_n_steps=5)
    trainer.fit(diffusion_pl, data_loader, test_loader)
