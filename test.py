import os
import argparse
import torch
from torch import nn

from dataset import get_loader
from Network import Network
from util import save_tensor_img
from config import Config


def main(args):
    # Init model
    config = Config()

    device = torch.device("cuda")
    model = Network()
    model = model.to(device)
    print('Testing with model {}'.format(args.ckpt))
    net_dict = torch.load(args.ckpt)

    model.to(device)
    model.load_state_dict(net_dict)

    model.eval()

    for testset in args.testsets.split('+'):
        print('Testing {}...'.format(testset))
        root_dir = './datasets/sod'
        if testset == 'CoCA':
            test_img_path = os.path.join(root_dir, 'images/CoCA')
            test_gt_path = os.path.join(root_dir, 'gts/CoCA')
            saved_root = os.path.join(args.pred_dir, 'CoCA')
        elif testset == 'CoSOD3k':
            test_img_path = os.path.join(root_dir, 'images/CoSOD3k')
            test_gt_path = os.path.join(root_dir, 'gts/CoSOD3k')
            saved_root = os.path.join(args.pred_dir, 'CoSOD3k')
        elif testset == 'CoSal2015':
            test_img_path = os.path.join(root_dir, 'images/CoSal2015')
            test_gt_path = os.path.join(root_dir, 'gts/CoSal2015')
            saved_root = os.path.join(args.pred_dir, 'CoSal2015')
        else:
            print('Unkonwn test dataset')
            print(args.dataset)

        test_loader = get_loader(
            test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

        for batch in test_loader:
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                scaled_preds = model(inputs,gts)[-1]

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            with torch.no_grad():
                 for inum in range(num):
                     subpath = subpaths[inum][0]
                     ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                     if config.db_output_refiner or (not config.refine and config.db_output_decoder):
                         res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True)
                     else:
                         res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True).sigmoid()
                     save_tensor_img(res, os.path.join(saved_root, subpath))


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',
                        default='Network',
                        type=str,
                        help="Options: '', ''")
    parser.add_argument('--testsets',
                        default='CoCA',
                        type=str,
                        help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")
    parser.add_argument('--size',
                        default=256,
                        type=int,
                        help='input size')
    parser.add_argument('--ckpt', default='./best_ep260_Smeasure0.908.pth', type=str, help='model folder')
    parser.add_argument('--pred_dir', default='./pred', type=str, help='Output folder')

    args = parser.parse_args()

    main(args)
