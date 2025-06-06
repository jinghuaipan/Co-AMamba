import os


class Config():
    def __init__(self) -> None:
        # Backbone
        self.bb = ['trans-pvt'][0]
        self.pvt_weights = ['./weights/pvt_v2_b2.pth', ''][0]
        # BN
        self.use_bn = self.bb not in ['cnn-vgg16']
        # Augmentation
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:3]
        self.val_measures = {
            'Emax': {'CoCA': 0.001, 'CoSOD3k': 0.001, 'CoSal2015': 0.001, 'MSRC': 0.001},
            'Smeasure': {'CoCA': 0.001, 'CoSOD3k': 0.001, 'CoSal2015': 0.001, 'MSRC': 0.001},
            'Fmax': {'CoCA': 0.001, 'CoSOD3k': 0.001, 'CoSal2015': 0.001, 'MSRC': 0.001},
            'MAE': {'CoCA': 0.001, 'CoSOD3k': 0.001, 'CoSal2015': 0.001, 'MSRC': 0.001},
        }
        # Components
        self.consensus = [''][0]
        self.dec_blk = ['ResBlk'][0]
        # Training
        self.batch_size = 16
        self.loadN = 2
        self.dec_att = ['', 'ASPP'][0]
        self.auto_pad = ['', 'fixed', 'adaptive'][0]
        self.optimizer = ['Adam', 'AdamW'][1]
        self.lr = 1e-4
        self.freeze = True
        self.lr_decay_epochs = [-20]    # Set to negative N to decay the lr in the last N-th epoch.
        self.forward_per_dataset = True
        losses = ['sal', 'contrast']
        self.loss = losses[:]
        self.cls_mask_operation = ['x', '+', 'c'][0]
        # Loss + Triplet Loss
        self.lambdas_sal_last = {
            'bce': 30 * 1,          # high performance
            'iou': 0.5 * 1,         # 0 / 255
            'ssim': 1 * 0,          # help contours
            'mse': 150 * 0,         # can smooth the saliency map
            'reg': 100 * 0,
            'triplet': 3 * 1,
        }

        self.db_output_decoder = False
        self.refine = False
        self.db_output_refiner = False
        # Triplet Loss
        self.triplet = ['_x5', 'mask'][:1]
        self.triplet_loss_margin = 0.1

        # Intermediate Layers
        self.lambdas_sal_others = {
            'bce': 0,
            'iou': 0.,
            'ssim': 0,
            'mse': 0,
            'reg': 0,
            'triplet': 0,
        }
        self.output_number = 1
        self.loss_sal_layers = 4              # used to be last 4 layers
        self.loss_cls_mask_last_layers = 1         # used to be last 4 layers
        if 'keep in range':
            self.loss_sal_layers = min(self.output_number, self.loss_sal_layers)
            self.loss_cls_mask_last_layers = min(self.output_number, self.loss_cls_mask_last_layers)
            self.output_number = min(self.output_number, max(self.loss_sal_layers, self.loss_cls_mask_last_layers))
            if self.output_number == 1:
                for cri in self.lambdas_sal_others:
                    self.lambdas_sal_others[cri] = 0
        self.conv_after_itp = False
        self.complex_lateral_connection = False

        self.loss_cls_mask_ratio_by_last_layers = 4 / self.loss_cls_mask_last_layers
        for loss_sal in self.lambdas_sal_last.keys():
            loss_sal_ratio_by_last_layers = 4 / (int(bool(self.lambdas_sal_others[loss_sal])) * (self.loss_sal_layers - 1) + 1)
            self.lambdas_sal_last[loss_sal] *= loss_sal_ratio_by_last_layers
            self.lambdas_sal_others[loss_sal] *= loss_sal_ratio_by_last_layers
        self.lambda_cls_mask = 2.5 * self.loss_cls_mask_ratio_by_last_layers
        self.lambda_cls = 3.
        self.intra_contrast = 1
        self.inter_contrast = 250

        # others
        self.self_supervision = False
        self.label_smoothing = False

        self.validation = True
        run_sh_file = [f for f in os.listdir('.') if 'go' in f and '.sh' in f] + [os.path.join('..', f) for f in os.listdir('..') if 'gco' in f and '.sh' in f]
        with open(run_sh_file[0], 'r') as f:
            lines = f.readlines()
            self.val_last = int([l.strip() for l in lines if 'val_last=' in l][0].split('=')[-1])
            self.save_step = int([l.strip() for l in lines if 'step=' in l][0].split('=')[-1])
