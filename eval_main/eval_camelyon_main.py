import sys
import argparse
sys.path.append('..')
import basic
from  basic.config import config_base
from  basic.utils import checkpoint
from  basic.utils import counter
from basic.utils import reflect
import time
import glob
import os
import time

import pdb


parser = argparse.ArgumentParser(description='huangxs eval_main')
parser.add_argument('-config', metavar='DIR', default='', help='config path')
parser.add_argument('-resume', '--resume', default=None, type=int, help='resume path')
parser.add_argument('-debug', default='false', help='debug mode')


def eval_main():
    """评测的主流程
    - 1. load config
    - 2. load model or resume
    - 3. timer
    - 4. train;test;validate # 在配置中选择是否运行
    """
    args = parser.parse_args()
    # load config
    config = config_base.ConfigBase(args.config)
    save_helper = checkpoint.CheckPoint(config)
    # 获取模型    
#    pdb.set_trace()
    model = reflect.get_model(config)
    train, validate, test, hard = get_instance(config, model)

    # timer
    time_counter = counter.Counter()
    time_counter.addval(time.time(), key='eval_main start')

    # eval
    time_counter.addval(time.time(), key='model load')
    hard_mining_times=0
    validation= config.get_config("test", 'run_this_module')
    if config.get_config("train", 'run_this_module') == True:
        train.train(model, hard_mining_times, save_helper,config,validation)
    # tain with hard_minning
    elif config.get_config('test','run_this_module') ==True:
        test.test(model, 0, hard_mining_times, save_helper)



def get_instance(_config, _model):
    train, validate, test, hard = None, None, None, None
    # get train/test instance by reflect
    if _config.get_config("train", 'run_this_module') == True:
        train = reflect.get_instance(_config, 'train')
        train.init_optimizer(_model)
    if _config.get_config("validate", 'run_this_module') == True:
        validate = reflect.get_instance(_config, 'validate')
    if _config.get_config("test", 'run_this_module') == True:
        test = reflect.get_instance(_config, 'test')
    if _config.get_config("hard", 'run_this_module') == True:
        hard = reflect.get_instance(_config, 'hard')
    return train, validate, test, hard


if __name__ == '__main__':
    eval_main()
