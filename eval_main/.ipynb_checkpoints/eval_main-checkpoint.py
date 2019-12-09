import sys

sys.path.append('../')
import argparse
import basic
from basic.config import config_base
from basic import eval_train
from basic import eval_test
from basic.utils import checkpoint
from basic.utils import counter
import time
import random
import pdb
import torchvision.models as models
from  tensorboardX import SummaryWriter

sys.path.append('./eval_train')
parser = argparse.ArgumentParser(description='huangxs eval_main')
parser.add_argument('-config', metavar='DIR', default='', help='config path')
parser.add_argument('-resume', '--resume', default=None, type=int, help='resume training epoch')
parser.add_argument('-debug', default='false', help='debug mode')


def eval_main():
    """完成一次配置的评测
    """
    args = parser.parse_args()
    # load config
    config = config_base.ConfigBase(args.config)
    save_helper = checkpoint.CheckPoint(config)
    # get train/test instance by reflect
    train = get_instance_by_key(config, 'train', save_helper)
    test = get_instance_by_key(config, 'test', save_helper)

    train_epoch_start = 0
    train_epoch_stop = train.config["epochs"]
    if args.resume:
        train_epoch_start = args.resume
    # timer
    time_counter = counter.Counter()
    time_counter.addval(time.time(), key='eval_main start')

    # eval
    model = train.load_model(0)
    time_counter.addval(time.time(), key='model load')
    writer = SummaryWriter(config.get_config('base','save_folder'))
    for i in range(train_epoch_start, train_epoch_stop):
        time_counter.addval(time.time(), key='eval epoch %d start' % i)
        train_acc, losses = train.train(model, i)
        writer.add_scalar('loss in each epoch',loss,i)
        if args.debug == 'false':
            test_acc = test.test(model, i)
            pass
        else:
            test_acc = {'avg_counter': counter.Counter(), 'epoch_acc_image': []}
        writer.add_scalar('accuracy in test',test_acc,i)
        writer.add_scalar('accuracy in train',train_acc,i)
        writer.add_scalar('accuracy in train',test_acc,i)
        # output pretty info
        save_helper.save(i, model, train_acc, losses, test_acc, time_counter)
        time_counter.addval(time.time(), key='eval epoch %d end' % i)
    save_helper.finish(train_epoch_start, train_epoch_stop)
    pass


def get_instance_by_key(config, config_key, save_helper):
    instance_config = config.get_config(config_key=config_key)
    _package = getattr(basic, instance_config['package_name'])
    _module = getattr(_package, instance_config['module_name'])
    _Class = getattr(_module, instance_config['class_name'])
    return _Class(instance_config, save_helper)


if __name__ == '__main__':
    eval_main()
