import os
import sys
import pdb


def check_mkdir(dir_name):
    """检验文件夹是否存在，不存在则创建"""
    #     import pdb; pdb.set_trace()
    if not os.path.exists(dir_name):
        # os.mkdir(dir_name)
        os.makedirs(dir_name, exist_ok=True)


def get_machine_name(save_folder):
    """获取当前机器的名称、标志号"""
    machine_name = 'no specific machine'
    machine_file = os.path.join(os.path.dirname(save_folder), 'machine_name.txt')
    if os.path.isfile(machine_file):
        f_machine_name = open(machine_file, 'r')
        machine_name = f_machine_name.read()
        f_machine_name.close()
    return machine_name
