import argparse
import json
import shutil
import os
from utils import ensure_dirs
class ConfigGAN(object):
    def __init__(self):

        # parse command line arguments
        parser, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.data_root = os.path.join('data', args.res)
        self.exp_dir = os.path.join(args.exp_name, args.res)
        self.log_dir = os.path.join(self.exp_dir, 'images')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        self.pred_dir = os.path.join(self.exp_dir, 'prediction')
        if args.res != '4080':
            self.pretrain_dir = os.path.join(args.exp_name, '4080/model/model_500.h5')
        if args.TOloss:
            if args.res != '200400':
                print('The resolution must be 200400!!!!!!!')
            else:
                self.TO_model = os.path.join(args.exp_name, '200400/model/model(BCE + 0.1bottleneck).h5')
                self.pred_dir = os.path.join(self.pred_dir, 'TOloss')
            
        ensure_dirs([self.log_dir, self.model_dir, self.pred_dir])

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--res', type=str, default="4080",
                            help="path to project folder where models and logs will be saved")
        parser.add_argument('--exp_name', type=str, default="GANTL", help="name of this experiment")

        parser.add_argument('--traintest', type=str, default="traintest", help="options: [traintest, test]")

        parser.add_argument('--batch_size', type=int, default=16, help="batch size")
        parser.add_argument('--load_point', type=int, default=500, help="The model load point")
        
        parser.add_argument('--n_testsamples', type=int, default=500, help="number of test samples to save")

        parser.add_argument('--n_epoch', type=int, default=500, help="total number of epochs to train")
        parser.add_argument('--save_step', type=int, default=50, help="save models every x epoch")
        parser.add_argument('--lr', type=float, default=2e-4, help="initial learning rate")
        
        parser.add_argument('--TOloss', type=bool, default=False, help="test of TO loss (res must be 200400)")
        args = parser.parse_args()
        return parser, args
