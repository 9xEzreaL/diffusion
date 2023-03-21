import os
import argparse
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from med_seg_diff_pytorch import Unet, MedSegDiff
from med_seg_diff_pytorch.dataset import ISICDataset, GenericNpyDataset
from accelerate import Accelerator
import skimage.io as io
import numpy as np



## Parse CLI arguments ##
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-od', '--output_dir', type=str, default="output", help="Output dir.")
    parser.add_argument('-ld', '--logging_dir', type=str, default="logs", help="Logging dir.")
    parser.add_argument('-mp', '--mixed_precision', type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to do mixed precision")
    parser.add_argument('-img', '--img_folder', type=str, default='ISBI2016_ISIC_Part3B_Training_Data',
                        help='The image file path from data_path')
    parser.add_argument('-csv', '--csv_file', type=str, default='ISBI2016_ISIC_Part3B_Training_GroundTruth.csv',
                        help='The csv file to load in from data_path')
    parser.add_argument('-sc', '--self_condition', action='store_true', help='Whether to do self condition')
    parser.add_argument('-ic', '--mask_channels', type=int, default=1, help='input channels for training (default: 3)')
    parser.add_argument('-c', '--input_img_channels', type=int, default=1,
                        help='output channels for training (default: 3)')
    parser.add_argument('-is', '--image_size', type=int, default=256, help='input image size (default: 128)')
    parser.add_argument('-dd', '--data_path', default='/media/ziyi/Dataset/TSE_DESS/TSE_DESS/train', help='directory of input image')
    parser.add_argument('-d', '--dim', type=int, default=64, help='dim (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs (default: 10000)')
    parser.add_argument('-bs', '--batch_size', type=int, default=6, help='batch size to train on (default: 8)')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps (default: 1000)')
    parser.add_argument('-ds', '--dataset', default='OAI_pain', help='Dataset to use')
    parser.add_argument('--save_every', type=int, default=100, help='save_every n epochs (default: 100)')
    parser.add_argument('--num_ens', type=int, default=5,
                        help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')
    parser.add_argument('--load_model_from', default=None, help='path to pt file to load from')
    parser.add_argument('--save_uncertainty', action='store_true',
                        help='Whether to store the uncertainty in predictions (only works for ensablmes)')

    return parser.parse_args()


def load_data(args):
    # Load dataset
    if args.dataset == 'ISIC':
        transform_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), ]
        transform_train = transforms.Compose(transform_list)
        dataset = ISICDataset(args.data_path, args.csv_file, args.img_folder, transform=transform_train, training=False,
                              flip_p=0.5)
    elif args.dataset == 'generic':
        transform_list = [transforms.ToPILImage(), transforms.Resize(args.image_size), transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        dataset = GenericNpyDataset(args.data_path, transform=transform_train, test_flag=True)
    elif args.dataset == 'oai':
        from med_seg_diff_pytorch.dataset import OAIDataset
        dataset = OAIDataset(args.data_path, 'test')
    elif args.dataset == 'pain':
        from med_seg_diff_pytorch.dataset import OAI_pain
        dataset = OAI_pain(args.data_path, 'test')
    else:
        raise NotImplementedError(f"Your dataset {args.dataset} hasn't been implemented yet.")

    ## Define PyTorch data generator
    training_generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    return training_generator
def to_8bit(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return img

def main():
    args = parse_args()
    inference_dir = os.path.join(args.output_dir, 'inference')
    os.makedirs(inference_dir, exist_ok=True)

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
    data_loader = load_data(args)

    diffusion = MedSegDiff(
        model,
        timesteps=args.timesteps
    ).cuda()

    if args.load_model_from is not None:
        save_dict = torch.load(args.load_model_from)
        diffusion.model.load_state_dict(save_dict['model_state_dict'])

    for (norm_img, pain_img, fname) in tqdm(data_loader):
        # pred = diffusion.sample(pain_img)
        pred_noise, preds = diffusion.sample(pain_img.float())
        preds = preds.cpu().detach()
        pred_noise = pred_noise.cpu().detach()
        for idx in range(preds.shape[0]):
            preds = to_8bit(np.array(preds))
            norm_img = to_8bit(np.array(norm_img))
            pain_img = to_8bit(np.array(pain_img))
            cat_preds = np.concatenate([preds[idx, 0, :, :], norm_img[idx, :, :], pain_img[idx, :, :]], 0)
            io.imsave(os.path.join(inference_dir, fname[idx]), np.expand_dims(cat_preds, 0))


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=0 python sample.py -ds pain --load_model_from output/pain-256-resize/checkpoints/state_dict_epoch_36_loss_0.19091334504774432.pt -dd '/media/ziyi/Dataset/OAI_pain/full' -is 256 --save_every 2