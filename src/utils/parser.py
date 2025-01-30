import yaml
import os
import argparse
from pathlib import Path
from argparse import ArgumentParser
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


from conf import Config

class Parser():
    def __init__(self, config_path:Path=None):
        
        self.parser = ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        self.config_path = config_path

        self.valid_optimizers = {
            'AdamW': 'AdamW optimizer, a variant of Adam Optimizer with weight decay.',
            'SGDM': 'Sthocastic Gradiant Descent with Momentum.',
            'LARS': 'Layer-wise Adaptive Rate Scaling optimizer, used for large-batch training.',
            'LAMB': 'Layer-wise Adaptive Moments for Batch training optimizer, used for large-batch training.'
        }
        self.valid_schedulers = {
            'CosineAnnealingLR': 'Cosine Annealing scheduler with learning rate decay.'
        }
        self.valid_dynamic = {
            'LossLS': 'Loss-based local step scheduler.',
            'AvgLossLS': 'Average loss-based local step scheduler.',
            'RevCosAnn': 'Reverse cosine annealing local step scheduler.',
            'Sigmoid': 'Sigmoid-based local step scheduler.',
            'ImprLS': 'Improvement-based local step scheduler.',
            'ALin': 'Ascending linear local step scheduler.',
            'DLin': 'Descending linear local step scheduler.',
            'ABin': 'Ascending binary local step scheduler.',
            'DBin': 'Descending binary local step scheduler.'
        }

        self.parser.add_argument('-c', '--config', type=Path, required=True if config_path is None else False, default=config_path, dest='CONFIG', help='Configuration file Path')
        self.parser.add_argument('--cpu', action='store_true', dest='CPU_F', help='Set CPU as Device')

        self.parser.add_argument('-lr', '--learning-rate', type=float, dest='LR', help='Overrides set Learning Rate')
        self.parser.add_argument('-bs', '--batch-size', type=int, dest='BS', help='Overrides set Batch Size')
        self.parser.add_argument('-ep', '--epochs', type=int, dest='EPOCHS', help='Overrides set Epochs')

        self.parser.add_argument('-opt', '--optimizer', type=str, dest='OPTIMIZER',choices= self.valid_optimizers.keys(), help='Overrides previously set optimizer. Valid choices:\n' + '\n'.join([f'{opt}: {desc}' for opt, desc in self.valid_optimizers.items()]))
        self.parser.add_argument('-sch', '--scheduler', type=str, dest='SCHEDULER', choices= self.valid_schedulers.keys(), help='Overrides previously set scheduler. Valid choices:\n' + '\n'.join([f'{opt}: {desc}' for opt, desc in self.valid_schedulers.items()]))

        self.parser.add_argument('-p', '--patience', type=int, dest='PATIENCE', help='Overrides previously set Patience')
        self.parser.add_argument('-wu', '--schewarmupduler', type=int, dest='WARMUP', help='Overrides previously set Warmup')
        self.parser.add_argument('-wd', '--weight-decay', type=float, dest='WD', help='Overrides previously set Weight Decay')

        self.parser.add_argument('-sm', '--slowmo-momentum', type=float, dest='SM', help='Overrides set SlowMo Momentum')
        self.parser.add_argument('-slr', '--slowmo-learning-rate', type=float, dest='SLR', help='Overrides set SlowMo Learning Rate')

        self.parser.add_argument('-nw', '--number-workers', type=int, dest='NW', help='Overrides set Number of Workers')
        self.parser.add_argument('-wss', '--worker-sync-step', type=int, dest='WSS', help='Overrides set workers Syncronised Steps')
        self.parser.add_argument('-wls', '--worker-local-step', type=int, dest='WLS', help='Overrides set workers Local Steps')
        self.parser.add_argument('-wbs', '--worker-batch-size', type=int, dest='WBS', help='Overrides set workers Batch Size')

        self.parser.add_argument('-dls', '--dynamic-strategy', type=str, dest='DWLS', choices= self.valid_dynamic.keys(), help='Overrides set dynamic local steps function. Valid choices:\n' + '\n'.join([f'{opt}: {desc}' for opt, desc in self.valid_dynamic.items()]))
        self.parser.add_argument('-dnl', '--dynamic-num-loss', type=int, dest='DNL', help='Overrides set number of losses used for dynamic local steps function')

        self.parser.add_argument('-P', '--pretrained', type=Path, dest='PRETRAINED', help='Path to the pretrained Model (Checkpoint)')
        self.parser.add_argument('-LC', '--load-checkpoint', action='store_true', dest='CHECKPOINT', help='Resume previous experiment')
        self.parser.add_argument('-TO', '--test-only', action='store_true', dest='TEST_F', help='Skip Training and do only Test')
        
        self.parser.add_argument('-en', '--experiment-name', type=str, dest='EXPERIMENT_NAME', help='Change Experiment')
        self.parser.add_argument('-v', '--version', type=float, dest='VERSION', help='Update Version')
        self.parser.add_argument('-o', '--output', type=Path, dest='OUTPUT', help="Set Output file path"  )

        

    def parse_args(self):
        self.args = self.parser.parse_args()

        device = 'cpu' if self.args.CPU_F else 'gpu'

        #load yaml file to the configuration
        with open(self.args.CONFIG) as file:
            d = yaml.safe_load(file)
            config = Config(**d)

        if self.args.LR is not None:
            config.model.learning_rate = self.args.LR

        if self.args.BS is not None:
            config.model.batch_size = self.args.BS

        if self.args.EPOCHS is not None:
            config.model.epochs = self.args.EPOCHS

        if self.args.OPTIMIZER is not None:
            config.model.optimizer = self.args.OPTIMIZER

        if self.args.SCHEDULER is not None:
            config.model.scheduler = self.args.SCHEDULER

        if self.args.WARMUP is not None:
            config.model.warmup = self.args.WARMUP

        if self.args.PATIENCE is not None:
            config.model.patience = self.args.PATIENCE
        
        if self.args.WD is not None:
            config.model.weight_decay = self.args.WD

        #optional Configuration in case of settes SlowMo
        if config.model.slowmo is not None:

            if self.args.SM is not None:
                config.model.slowmo.momentum = self.args.SM

            if self.args.SLR is not None:
                config.model.slowmo.learning_rate = self.args.SLR
        
        elif self.args.SM is not None or self.args.SLR is not None:
            raise argparse.ArgumentError(None, "It's not possibe to modify SlowMo Parameters if SlowMo is not Setin the configuration File")

        if self.args.NW is not None:
                config.model.num_workers = self.args.NW

        #optional Configuration in case of distributed training
        if config.model.num_workers > 0:

            if self.args.WSS is not None:
                config.model.work.sync_steps = self.args.WSS
            
            if self.args.WLS is not None:
                config.model.work.local_steps = self.args.WLS
            
            if self.args.WBS is not None:
                config.model.work.batch_size = self.args.WBS
            
            if config.model.work.dynamic is not None:
                if self.args.DWLS is not None:
                    config.model.work.dynamic.strategy = self.args.DWLS
                    # parameter applied only when a specific strategy has been chosen
                    if self.args.DNL is not None:
                        config.model.work.dynamic.n_losses = self.args.DNL

            elif self.args.DWLS is not None or self.args.DNL is not None:
                raise argparse.ArgumentError(None, "It's not possibe to modify Dynamic Parameters if Dynamic is not Set in the configuration File")
            
        else:
            config.model.num_workers = 0
            config.model.work = None


        if self.args.PRETRAINED is not None:
            config.model.pretrained = self.args.PRETRAINED

        if self.args.CHECKPOINT:
            config.experiment.resume = self.args.CHECKPOINT

        if self.args.TEST_F:
            if config.model.pretrained:
                config.experiment.test_only = self.args.TEST_F
            else:
                raise argparse.ArgumentError(None, "It's not possibe to only test a model whithout setting the pretrained model file")
        
        if self.args.EXPERIMENT_NAME is not None:
            config.experiment.name = self.args.EXPERIMENT_NAME

        if self.args.VERSION is not None:
            config.experiment.version = self.args.VERSION
        
        if self.args.OUTPUT is not None:
            config.experiment.output = self.args.OUTPUT
        
        # Set checkpoint directory based on the experiment Name
        config.experiment.checkpoint_dir = os.path.join(config.experiment.checkpoint_dir,config.experiment.name)
        os.makedirs(config.experiment.checkpoint_dir, exist_ok=True)

        return config, device
