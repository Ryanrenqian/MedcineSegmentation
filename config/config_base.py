import os
import sys
import time
import json
from  basic.utils.logs import Log
from  basic.utils import timeutil
from  basic.utils import file


class ConfigBase(object):
    """
    以单词实验为维度，一个配置表内包含了在数据集上的实验及其结果
    一个数据集建议使用多个config文件配置实验，不要都挤在一个config里面
    """

    def __init__(self, config_path):
        self.config_ids = []
        self.key_to_config = {}
        self.config_list = []
        self.config_path = config_path
        f_config = open(config_path, 'r', encoding='utf-8')
        _content = f_config.read()
        f_config.close()
        self.config = json.loads(_content)

        #self.update_config(self.config_path)

    def get_config(self, *args):
        """
        获取配置信息
        :return:
        """
        config_tree = self.config
        for arg in args:
            if arg in config_tree.keys():
                config_tree = config_tree[arg]
            else:
                config_tree = None
        return config_tree

    def update_config(self):
        """ 更新当前运行的配置"""
        last_run_date = timeutil.get_timestr()
        last_run_machine = os.path.join(self.config['base']['save_folder'],"config")
        # 如果是恢复运行,使用之前的路径
        self.config['base']['last_run_date'] = last_run_date

        self.config['base']['last_run_machine'] = last_run_machine
        f_config = open(last_run_machine, 'w', encoding='utf-8')
        f_config.write(json.dumps(self.config, ensure_ascii=False, indent=2))
        f_config.close()
