import sys

sys.path.append('../')
import argparse
import basic
from  basic.config import config_base
from  basic.utils import checkpoint
from  basic.utils import counter
import basic.utils.reflect as reflect
import time
import glob
import os
import time

import pdb

sys.path.append('./eval_train')
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
    if save_helper.model_state != None:
        model.load_state_dict(save_helper.model_state)

    train, validate, test, hard = get_instance(config, model)
    train_epoch_start = 0 if config.get_config("train", 'start_epoch') == None else config.get_config("train",
                                                                                                      'start_epoch')
    train_epoch_stop = config.get_config("train", 'total_epoch')
    if args.resume:
        # TODO 增加resume的方式
        pass
    # timer
    time_counter = counter.Counter()
    time_counter.addval(time.time(), key='eval_main start')

    # eval
    time_counter.addval(time.time(), key='model load')

    for hard_mining_times in range(10):
        for i in range(train_epoch_start, train_epoch_stop):
            time_counter.addval(time.time(), key='eval epoch %d start' % i)
            if config.get_config("train", 'run_this_module') == True:
                train_acc, losses = train.train(model, i, hard_mining_times, save_helper)

            if config.get_config("validate", 'run_this_module') == True:
                validate.validate(model, i, hard_mining_times, save_helper)

            if config.get_config("test", 'run_this_module') == True:
                test.test(model, i, hard_mining_times, save_helper)

            # if i > 0 and i % 11 == 0 and config.get_config("hard", 'run_this_module') == True:
            if config.get_config("hard", 'run_this_module') == True:
                print('\n', i)
                hard_acc = hard.hard(model, i, hard_mining_times, save_helper)
                # 挖掘完后重新开始训练和挖掘
                # hard.reload_data()
                # train.reload_data()
                # 重置学习率，但保留之前的模型
                # train.init_optimizer(model)
                break

            # output pretty info,保存模型
            # save_helper.save(i, model, train_acc, losses, test_acc, time_counter)
            time_counter.addval(time.time(), key='eval epoch %d end' % i)
            save_helper.finish(i, train_epoch_start, train_epoch_stop)
        pass


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
#     pdb.set_trace()
    eval_main()
