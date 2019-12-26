"""save checkpoint information"""
import os
import shutil
import torch
import pdb

from  ..utils import logs
from  ..utils import file
from  ..utils import timeutil
import json

best_acc = 0


class CheckPoint(object):
    def __init__(self, config):
        super(CheckPoint, self).__init__()
        self.config = config
        self.model_state = None
        self.train_acc_list = []
        self.losses_list = []
        self.test_acc_list = []
        self.best_acc = 0
        self.best_epoch = 0
        self.save_folder = self.config.get_config('base', 'save_folder')
        self.log = logs.Log(os.path.join(self.save_folder, 'log.txt'))

        # 把运行的config备份到结果文件夹
        if self.config.get_config('base', 'resume_checkpoint') != None and len(
                self.config.get_config('base', 'resume_checkpoint')) > 0:
            self.resume_checkpoint(self.config.get_config('base', 'resume_checkpoint'),
                                   self.config.get_config('base', 'resume_only_model'))
        config.update_config()

    def save_epoch_pred(self, workspace,epoch_image_results, txt_name):
        file.check_mkdir(workspace)
        save_name = os.path.join(workspace, txt_name)
        print('\nsave file to %s' % save_name)
        f = open(save_name, 'w')
        f.writelines(json.dumps(epoch_image_results, indent=4))
        f.close()

# 这里用来保存hard example
    def save_hard_example(self,filename,records):
        with open(filename,'w')as f:
            f.writelines(records)

    def save_epoch_model(self, workspace,epoch, run_type, acc, losses, model, iteration):
        """
        保存单轮的运行结果，但不保存模型，模型只保留best和最后一个
        :param epoch:
        :param run_type: 保存的类型标签
        :param acc:
        :param losses:
        :return:
        """
        save_path = os.path.join(workspace,'models')
        os.system(f'mkdir -p {save_path}')
        save_name = os.path.join(save_path,f'epoch_{epoch}_{run_type}_acc_losses.pth')
        # 保存中间结果
        torch.save({"iteration":iteration,
                    "epoch": epoch,
                    "acc": acc,
                    "losses": losses}, save_name)
        # 保存模型参数
        save_model_name = os.path.join(save_path,f'epoch_{epoch}_{run_type}_model.pth')
        torch.save({"epoch": epoch,
                    "model_state": model.state_dict()}, save_model_name)

    def save(self, epoch, model, train_acc, losses, test_acc, iteration=None,time_counter=None):
        """
        保存中间模型结果和最佳模型,模型只保留两个。这是对整个epochs进行操作的
        - 当前的和最佳的
        - 过程参数都保存下来，包括failure example
        :param epoch:
        :param model:当前模型
        :param train_acc:当前train acc
        :param losses: 当前epoch losses
        :param test_acc: 当前 epoch模型的test acc
        :return:
        """
        # 如果是best，更新best model 的check point
        is_best_acc = False
        if self.best_acc < test_acc['avg_counter'].avg:
            self.best_acc = test_acc['avg_counter'].avg
            self.best_epoch = epoch
            is_best_acc = True

        # 存储每个epoch 的 过程数据
        self.train_acc_list.append(train_acc)
        self.losses_list.append(losses)
        self.test_acc_list.append(test_acc)

        # 正常的epoch只保留过程参数
        save_epoch_path = os.path.join(self.save_folder, 'epoch.pth')
        #         pdb.set_trace()
        if epoch % self.config.get_config('base', 'checkpoint_save_freq') == 0:
            torch.save({"epoch": epoch,
                        "config": self.config,
                        "train_acc_list": self.train_acc_list,
                        "test_acc_list": self.test_acc_list,
                        "losses_list": self.losses_list,
                        "best_acc": self.best_acc,
                        "best_epoch": self.best_epoch,
                        "time_counter": time_counter},
                       save_epoch_path)
            self.log.info('    save epoch.pth:%s' % save_epoch_path)

        # 间隔epoch时，保留模型
        if epoch % self.config.get_config('base', 'epoch_save_freq') == 0:
            save_checkpoint_path = os.path.join(self.save_folder, 'checkpoint.pth')
            torch.save({"epoch": epoch,
                        "config": self.config,
                        "train_acc_list": self.train_acc_list,
                        "test_acc_list": self.test_acc_list,
                        "losses_list": self.losses_list,
                        "best_acc": self.best_acc,
                        "best_epoch": self.best_epoch,
                        "model_state": model.state_dict(),
                        'iteration':iteration},
                       save_checkpoint_path)
            self.log.info('    save checkpoint.pth:%s' % save_checkpoint_path)

        # 如果本次epoch结果最佳
        if is_best_acc:
            save_best_acc_path = os.path.join(self.save_folder,'best_acc.pth')
            torch.save({"epoch": epoch,
                        "config": self.config,
                        "train_acc_list": self.train_acc_list,
                        "test_acc_list": self.test_acc_list,
                        "losses_list": self.losses_list,
                        "best_acc": self.best_acc,
                        "best_epoch": self.best_epoch,
                        "model_state": model.state_dict()},
                       save_best_acc_path)
            self.log.info('save best_acc.pth:%s' % save_best_acc_path)
        # 保留每次epoch的failure example

    def finish(self, epoch, epoch_start, epoch_stop):
        finish_path = os.path.join(self.save_folder, 'last_finish_%d_%d_%d.pth' % (epoch, epoch_start, epoch_stop))
        torch.save({"epoch_start": epoch_start, "epoch_stop": epoch_stop}, finish_path)

    def resume_checkpoint(self, resume_path, resume_only_model=False):
        self.log.info('resume from  checkpoint :%s' % resume_path)
        epoch_checkpoint = torch.load(resume_path)
#         pdb.set_trace()
        self.epoch = epoch_checkpoint['epoch']
        if not resume_only_model:
            self.train_acc_list = epoch_checkpoint['train_acc_list']
            self.losses_list = epoch_checkpoint['losses_list']
            self.test_acc_list = epoch_checkpoint['test_acc_list']
            self.best_acc = epoch_checkpoint['best_acc']
            self.best_epoch = epoch_checkpoint['best_epoch']
        self.model_state = epoch_checkpoint['model_state']

        resume_info = os.path.join(self.save_folder, 'resume_epoch_%d.txt' % (self.epoch))
        f = open(resume_info, 'w')
        f.write(json.dumps({"resume_path": resume_path}, ensure_ascii=False, indent=2))
        f.close()

