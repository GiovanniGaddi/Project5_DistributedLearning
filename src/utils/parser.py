import yaml
import os
from pathlib import Path
from argparse import ArgumentParser
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


from conf import Config

class Parser():
    def __init__(self, config_path:Path=None):
        
        self.parser = ArgumentParser()
        self.config_path = config_path

        self.parser.add_argument('-c', '--config', type=Path, required=True if config_path is None else False, default=config_path, dest='CONFIG', help='Configuration file Path')
        self.parser.add_argument('--cpu', action='store_true', dest='CPU_F', help='Set CPU as Device')

        self.parser.add_argument('-lr', '--learning_rate', type=float, dest='LR', help='Overload setted Learning Rate')
        self.parser.add_argument('-bs', '--batch_size', type=int, dest='BS', help='Overload setted Batch Size')
        self.parser.add_argument('-ep', '--epochs', type=int, dest='EPOCHS', help='Overload setted Epochs')

        self.parser.add_argument('-P', '--pretrained', type=Path, dest='PRETRAINED', help='Path to the pretrained Model (Checkpoint)')
        self.parser.add_argument('-LC', '--load_checkpoint', action='store_true', dest='CHECKPOINT', help='Resume previous experiment')
        self.parser.add_argument('-T', '--test', action='store_true', dest='TEST_F', help='Skip Training and do only Test')
        
        self.parser.add_argument('-en', '--experiment_name', type=str, dest='EXPERIMENT_NAME', help='Change Experiment')
        self.parser.add_argument('-v', '--version', type=float, dest='VERSION', help='Update Version')

        self.parser.add_argument('-opt', '--optimizer', type=str, dest='OPTIMIZER', help='Overrides previously set optimizer')
        self.parser.add_argument('-sch', '--scheduler', type=str, dest='SCHEDULER', help='Overrides previously set scheduler')

    def parse_args(self):
        self.args = self.parser.parse_args()

        device = 'cpu' if self.args.CPU_F else 'gpu'

        with open(self.args.CONFIG) as file:
            d = yaml.safe_load(file)
            config = Config(**d)

        if self.args.LR is not None:
            config.model.learning_rate = self.args.LR

        if self.args.BS is not None:
            config.model.batch_size = self.args.BS

        if self.args.EPOCHS is not None:
            config.model.epochs = self.args.EPOCHS

        if self.args.PRETRAINED is not None:
            config.model.pretrained = self.args.PRETRAINED

        if self.args.CHECKPOINT is not None:
            config.experiment.resume = self.args.CHECKPOINT

        if self.args.TEST_F is not None:
            config.model.test = self.args.TEST_F
        
        if self.args.EXPERIMENT_NAME is not None:
            config.experiment.experiment_name = self.args.EXPERIMENT_NAME

        if self.args.VERSION is not None:
            config.experiment.version = self.args.VERSION
        
        if self.args.OPTIMIZER is not None:
            config.model.optimizer = self.args.OPTIMIZER

        if self.args.SCHEDULER is not None:
            config.model.scheduler = self.args.SCHEDULER
        
        config.checkpoint.dir = os.path.join(config.checkpoint.dir,config.experiment.name)
        os.makedirs(config.checkpoint.dir, exist_ok=True)

        return config, device