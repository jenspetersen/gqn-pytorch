import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from trixi.util import Config, GridSearch


class ConvModule(nn.Module):
    """Utility Module for more convenient weight initialization"""

    conv_types = (nn.Conv1d,
                  nn.Conv2d,
                  nn.Conv3d,
                  nn.ConvTranspose1d,
                  nn.ConvTranspose2d,
                  nn.ConvTranspose3d)

    @classmethod
    def is_conv(cls, op):

        if type(op) == type and issubclass(op, cls.conv_types):
            return True
        elif type(op) in cls.conv_types:
            return True
        else:
            return False

    def __init__(self, *args, **kwargs):

        super(ConvModule, self).__init__(*args, **kwargs)

    def init_weights(self, init_fn, *args, **kwargs):

        class init_(object):

            def __init__(self):
                self.fn = init_fn
                self.args = args
                self.kwargs = kwargs

            def __call__(self, module):
                if ConvModule.is_conv(type(module)):
                    module.weight = self.fn(module.weight, *self.args, **self.kwargs)

        _init_ = init_()
        self.apply(_init_)

    def init_bias(self, init_fn, *args, **kwargs):

        class init_(object):

            def __init__(self):
                self.fn = init_fn
                self.args = args
                self.kwargs = kwargs

            def __call__(self, module):
                if ConvModule.is_conv(type(module)) and module.bias is not None:
                    module.bias = self.fn(module.bias, *self.args, **self.kwargs)

        _init_ = init_()
        self.apply(_init_)


def get_default_experiment_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="Working directory for experiment.")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to a config file.")
    parser.add_argument("-v", "--visdomlogger", action="store_true", help="Use visdomlogger.")
    parser.add_argument("-dc", "--default_config", type=str, default="DEFAULTS", help="Select a default Config")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume from")
    parser.add_argument("-ir", "--ignore_resume_config", action="store_true", help="Ignore Config in experiment we resume from.")
    parser.add_argument("--test", action="store_true", help="Run test instead of training")
    parser.add_argument("--grid", type=str, help="Path to a config for grid search")
    parser.add_argument("-s", "--skip_existing", action="store_true", help="Skip configs fpr which an experiment exists")
    parser.add_argument("-m", "--mods", type=str, nargs="+", default=None, help="Mods are Config stubs to update only relevant parts for a certain setup.")

    return parser


def run_experiment(experiment, configs, args, mods=None, **kwargs):

    config = Config(file_=args.config) if args.config is not None else Config()
    config.update_missing(configs[args.default_config])
    if args.mods is not None:
        for mod in args.mods:
            config.update(mods[mod])
    config = Config(config=config, update_from_argv=True)

    # GET EXISTING EXPERIMENTS TO BE ABLE TO SKIP CERTAIN CONFIGS
    if args.skip_existing:
        existing_configs = []
        for exp in os.listdir(args.base_dir):
            try:
                existing_configs.append(Config(file_=os.path.join(args.base_dir, exp, "config", "config.json")))
            except Exception as e:
                pass

    if args.grid is not None:
        grid = GridSearch().read(args.grid)
    else:
        grid = [{}]

    for combi in grid:

        config.update(combi)

        if args.skip_existing:
            skip_this = False
            for existing_config in existing_configs:
                if existing_config.contains(config):
                    skip_this = True
                    break
            if skip_this:
                continue

        loggers = {}
        if args.visdomlogger:
            loggers["visdom"] = ("visdom", {}, 1)

        exp = experiment(config=config,
                         base_dir=args.base_dir,
                         resume=args.resume,
                         ignore_resume_config=args.ignore_resume_config,
                         loggers=loggers,
                         **kwargs)

        if not args.test:
            exp.run()
        else:
            exp.run_test()


def set_seeds(seed, cuda=True):

    if not hasattr(seed, "__iter__"):
        seed = (seed, seed, seed)
    np.random.seed(seed[0])
    torch.manual_seed(seed[1])
    if cuda: torch.cuda.manual_seed_all(seed[2])
