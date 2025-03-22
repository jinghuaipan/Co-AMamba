import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from util import Logger, AverageMeter, set_seed
import os
import argparse
from dataset import get_loader
import pytorch_toolbelt.losses as PTL
import torch.nn.functional as F

from config import Config
from loss import saliency_structure_consistency, SalLoss
from util import generate_smoothed_gt

from Network import Network
from train_val.evaluations.dataloader import EvalDataset
from train_val.evaluations.evaluator import Eval_thread
from train_val.dataset import get_loaders
from train_val.util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed

# Parameter from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model',
                    default='Network',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--resume',
                    default='./best_ep277_Smeasure0.905.pth',
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='seg_duts',
                    type=str,
                    help="Options: 'seg_duts'")
parser.add_argument('--testsets',
                    default='CoSal2015+CoCA+CoSOD3k',
                    type=str,
                    help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")
parser.add_argument('--size',
                    default=256,
                    type=int,
                    help='input size')

# parser.add_argument('--ckpt_dir',
# default='./ckpt',
# help='Temporary folder')


parser.add_argument('--save_root', default='./data/pred', type=str, help='Output folder')
parser.add_argument('--tmp', default='./weight/', help='Temporary folder')

args = parser.parse_args()

config = Config()

# Prepare dataset
root_dir = './trainsets/sod/'
if 'seg_duts' in args.trainset.split('+'):
    train_img_path = os.path.join(root_dir, 'images/seg_duts')
    train_gt_path = os.path.join(root_dir, 'gts/seg_duts')
    train_loader = get_loader(
        train_img_path,
        train_gt_path,
        args.size,
        1,
        max_num=config.batch_size,
        istrain=True,
        shuffle=True,
        num_workers=8,
        pin=True
    )


else:
    print('Unkonwn train dataset')
    print(args.dataset)

test_loaders = {}
for testsets in args.testsets.split('+'):
    test_loader = get_loader(
        os.path.join('./datasets/sod', 'images', testsets), os.path.join('./datasets/sod', 'gts', testsets),
        args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True
    )
    test_loaders[testsets] = test_loader

# if config.rand_seed:
#     set_seed(config.rand_seed)

# make dir for ckpt
# os.makedirs(args.ckpt_dir, exist_ok=True)
if args.tmp is not None:
    os.makedirs(args.tmp, exist_ok=True)
else:
    print("Error: tmp is not set")
# Init log file
logger = Logger(os.path.join(args.tmp, "log.txt"))
logger_loss_file = os.path.join(args.tmp, "log_loss.txt")
logger_loss_idx = 1

# Init model
device = torch.device("cuda")  # CUDA

model = Network().to(device)

# Setting optimizer
if config.optimizer == 'AdamW':
    optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
elif config.optimizer == 'Adam':
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[lde if lde > 0 else args.epochs + lde for lde in config.lr_decay_epochs],
    gamma=0.1
)

# Why freeze the backbone?...
if config.freeze:
    for key, value in model.named_parameters():
        if 'bb.' in key:
            value.requires_grad = False

# log model and optimizer params
# logger.info("Model details:")
# logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
logger.info(lr_scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
sal_loss = SalLoss()


def compute_cos_dis(x_sup, x_que):
    x_sup = x_sup.reshape(x_sup.size()[0], x_sup.size()[1], -1)
    x_que = x_que.reshape(x_que.size()[0], x_que.size()[1], -1)

    x_que_norm = torch.norm(x_que, p=2, dim=1, keepdim=True)
    x_sup_norm = torch.norm(x_sup, p=2, dim=1, keepdim=True)

    x_que_norm = x_que_norm.permute(0, 2, 1)
    x_qs_norm = torch.matmul(x_que_norm, x_sup_norm)

    x_que = x_que.permute(0, 2, 1)

    x_qs = torch.matmul(x_que, x_sup)
    x_qs = x_qs / (x_qs_norm + 1e-5)
    return x_qs


def sclloss(x, xt, xb):
    cosc = (1 + compute_cos_dis(x, xt)) * 0.5
    cosb = (1 + compute_cos_dis(x, xb)) * 0.5
    loss = -torch.log(cosc + 1e-5) - torch.log(1 - cosb + 1e-5)
    return loss.sum()


for testset in ['CoSal2015']:
    if testset == 'CoSal2015':
        test_img_path = './datasets/sod/images/CoSal2015/'
        test_gt_path = './datasets/sod/gts/CoSal2015/'

        saved_root = os.path.join(args.save_root, 'CoSal2015')

        # elif testset == 'CoCA':
        # test_img_path = './datasets/sod/images/CoCA/'
        # test_gt_path = './datasets/sod/gts/CoCA/'

        # saved_root = os.path.join(args.save_root, 'CoCA')

        test_loader = get_loaders(
            test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)
        
validation = True

def main():
    # Optionally resume from a checkpoint
    val_measures = []
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model.load_state_dict(torch.load(args.resume))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    print(args.epochs)
    for epochs in range(args.start_epoch, args.epochs):
        train_loss = train(epochs)
        if validation:
            measures = validate(model, test_loader, args.testsets)
            val_measures.append(measures)
            print(
                'Validation: Smeasure on CoSal2015 for epoch-{} is {:.3f}. Best epoch is epoch-{} with Smeasure {:.3f}'.format(
                    epochs, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
                    np.max(np.array(val_measures)[:, 0]))
            )
            # Save checkpoint
        save_checkpoint(
            {
                'epochs': epochs + 1,
                'state_dict': model.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            },
            path=args.tmp)
        if validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                best_weights_before = [os.path.join(args.tmp, weight_file) for weight_file in
                                       os.listdir(args.tmp) if 'best_' in weight_file]
                for best_weight_before in best_weights_before:
                    os.remove(best_weight_before)
                torch.save(model.state_dict(),
                           os.path.join(args.tmp, 'best_ep{}_Smeasure{:.3f}.pth'.format(epochs, measures[0])))
        if (epochs + 1) % 1 == 0 or epochs == 0:
            torch.save(model.state_dict(), args.tmp + '/model-' + str(epochs + 1) + '.pth')

        # if epochs > 1:
        # torch.save(model.state_dict(), args.tmp + '/model-' + str(epochs) + '.pth')
    # dcfmnet_dict = model.dcfmnet.state_dict()
    # torch.save(dcfmnet_dict, os.path.join(args.tmp, 'final.pth'))


def train(epoch):
    loss_log = AverageMeter()
    loss_log_triplet = AverageMeter()
    global logger_loss_idx
    FL = PTL.BinaryFocalLoss()
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        cls_gts = torch.LongTensor(batch[-1]).to(device)
        gts_neg = torch.full_like(gts, 0.0)
        gts_cat = torch.cat([gts, gts_neg], dim=0)
        # H, W = inputs.shape[-2:]
        return_values = model(inputs, gts)
        scaled_preds = return_values[0]
        pred_contrast = return_values[5]
        norm_features = None
        # if {'sal', 'contrast'} == set(config.loss):
        # scaled_preds, pred_contrast = return_values[:2]
        ##if config.GCAM_metric:
        ##norm_features = return_values[-1]
        ##scaled_preds = scaled_preds[-min(config.loss_sal_layers+int(bool(config.refine)), 4+int(bool(config.refine))):]

        # Tricks
        ##if config.GCAM_metric:
        ##loss_sal, loss_triplet = sal_loss(scaled_preds, gts, norm_features=norm_features, labels=cls_gts)
        ##else:
        loss_scl = sclloss(return_values[1], return_values[2], return_values[3])
        loss_sal = sal_loss(scaled_preds, gts) + 0.0001 * loss_scl

        if config.label_smoothing:
            loss_sal = 0.5 * (loss_sal + sal_loss(scaled_preds, generate_smoothed_gt(gts)))
        if config.self_supervision:
            H, W = inputs.shape[-2:]
            images_scale = F.interpolate(inputs, size=(H // 4, W // 4), mode='bilinear', align_corners=True)
            sal_scale = model(images_scale)[0][-1]
            atts = scaled_preds[-1]
            sal_s = F.interpolate(atts, size=(H // 4, W // 4), mode='bilinear', align_corners=True)
            loss_ss = saliency_structure_consistency(sal_scale.sigmoid(), sal_s.sigmoid())
            loss_sal += loss_ss * 0.3

        # Loss
        # since there may be several losses for sal, the lambdas for them (lambdas_sal) are inside the loss.py
        ##loss = loss_sal * 1.0
        loss = 0
        # since there may be several losses for sal, the lambdas for them (lambdas_sal) are inside the loss.py
        loss_sal = loss_sal * 1
        loss += loss_sal
        if 'contrast' in config.loss:
           loss_contrast = FL(pred_contrast, gts_cat) * config.lambda_contrast#([128, 1, 256, 256]) ([64, 1, 256, 256])
           loss += loss_contrast
        if config.forward_per_dataset:
            loss_log.update(loss, inputs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with open(logger_loss_file, 'a') as f:
            f.write('step {}, {}\n'.format(logger_loss_idx, loss))
        logger_loss_idx += 1
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Logger
        if batch_idx % 20 == 0:
            # NOTE: Top2Down; [0] is the grobal slamap and [5] is the final output
            info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}]'.format(epoch, args.epochs, batch_idx, len(train_loader))
            info_loss = 'Train Loss: loss_sal: {:.3f}'.format(loss_sal)
            if 'contrast' in config.loss:
                info_loss += ', loss_contrast: {:.3f}'.format(loss_contrast)
            info_loss += ', Loss_total: {loss.val:.3f} ({loss.avg:.3f})  '.format(loss=loss_log)
            logger.info(''.join((info_progress, info_loss)))
    info_loss = '@==Final== Epoch[{0}/{1}]  Train Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=loss_log)
    # if config.GCAM_metric:
    # info_loss += 'Triplet Loss: {loss.avg:.3f}  '.format(loss=loss_log_triplet)
    logger.info(info_loss)
    # if epoch % 1 == 0:
    # torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'ep{}_checkpoint.pth'.format(epoch)))#

    return loss_log.avg


def validate(model, test_loader, testsets):
    model.eval()

    testsets = testsets.split('+')
    measures = []
    for testset in testsets[:1]:
        print('Validating {}...'.format(testset))
        # test_loader = test_loaders[testset]

        saved_root = os.path.join(args.save_root, testset)

        for batch in test_loader:
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                H, W = inputs.shape[-2:]
                scaled_preds = model(inputs, gts)[-1].sigmoid()

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                align_corners=True)
                save_tensor_img(res, os.path.join(saved_root, subpath))

        eval_loader = EvalDataset(
            saved_root,  # preds
            os.path.join('./datasets/sod/gts', testset)  # GT
        )
        evaler = Eval_thread(eval_loader, cuda=False)
        # Use s_measure for validation
        s_measure = evaler.Eval_Smeasure()
        if s_measure > config.val_measures['Smeasure']['CoSal2015']:
            # TODO: evluate others measures if s_measure is very high.
            # e_max = evaler.Eval_Emeasure().max().item()
            # f_max = evaler.Eval_fmeasure().max().item()
            MAE = evaler.Eval_mae()

            print('MAE: {:.3f}'.format(MAE))
        measures.append(s_measure)

    model.train()
    return measures


if __name__ == '__main__':
    main()
