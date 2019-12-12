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

def get_save_dir(base_dir, name, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        save_dir = os.path.join(base_dir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')
