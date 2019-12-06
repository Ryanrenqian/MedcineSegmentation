import os
import sys
import glob
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='generate patch points of WSI')
    parser.add_argument('-of', '--otsu_folder',
                        default='/root/workspace/renqian/0929/prepare_data/wsi_otsu_save/train_resize_64',
                        type=str, help='otsu图片列表')
    parser.add_argument('-gf', '--gt_mask_folder',
                        default='/root/workspace/huangxs/prepare_data/16/wsi_mask/train_64',
                        type=str, help='由标注生成的mask')
    parser.add_argument('-sp', '--save_path',
                        default='/root/workspace/renqian/1115/patch_list/',
                        type=str, help='生成的patch list路径')
    parser.add_argument('-ds', '--downsample',
                        default=64,
                        type=int, help='生成的patch list路径')
    args = parser.parse_args()
    return args

def generate_patch_from_slide(slide_basename, normal_file,tumor_file, otsu_dict, gt_mask_dict, down_sample=64):
    """
    边扫描变生成新的dataset
    """

    slide_otsu = otsu_dict[slide_basename]
    x, y = np.where(slide_otsu > 0)
    xy = [[x[i], y[i]] for i in range(len(x))]

    tumor_count, normal_count = 0, 0
    slide_is_tumor = False
    if slide_basename in gt_mask_dict.keys():
        slide_gt_mask = gt_mask_dict[slide_basename]
        slide_is_tumor = True
    for i in range(len(xy)):
        _x, _y = xy[i]
        # 取patch的时候，以_x,_y像素块的中心为中心。偏差大约32像素点
        level0_x, level0_y = int((_x+0.5) * down_sample), int((_y+0.5) * down_sample)
        if slide_is_tumor:
            if slide_gt_mask[_x, _y] > 0:
                tumor_file.write('%s.tif_%04d_%04d\n' % (slide_basename, level0_x, level0_y))
                tumor_count += 1
            else:
                normal_file.write('%s.tif_%04d_%04d\n' % (slide_basename, level0_x, level0_y))
                normal_count += 1
        else:
            normal_file.write('%s.tif_%04d_%04d\n' % (slide_basename, level0_x, level0_y))
            normal_count +=1
    return tumor_count,normal_count

def skip_slide(slide_name):
    skip_list = ['normal_86', 'normal_144', 'test_049', 'test_114']
    for skip_name in skip_list:
        if skip_name in slide_name:
            return True
    return False

def main():
    args = get_args()
    otsu_folder = args.otsu_folder
    gt_mask_folder = args.gt_mask_folder
    save_path = args.save_path
    downsample = args.downsample
    otsu_dict = {}
    otsu_list = glob.glob(os.path.join(otsu_folder, '*.npy'))
    otsu_list.sort()
    total_point = 0
    tumor_count,normal_count=0, 0
    normal_file=os.path.join(save_path,'train_normal.list')
    f_n = open(normal_file,'w')
    tumor_file=os.path.join(save_path,'train_tumor.list')
    f_t = open(tumor_file,'w')
    # reading otsu
    for otsu in otsu_list:
        # 剔除部分数据，tumor_114
        if skip_slide(otsu):
            continue
        _basename = os.path.basename(otsu).split('_resize')[0]
        otsu_dict[_basename] = np.load(otsu)
        x, y = np.where(otsu_dict[_basename] > 0)
        total_point += len(x)
    print(f'read OSTU: {total_point}')
    # raading gt mask
    gt_mask_list = glob.glob(os.path.join(gt_mask_folder, '*.npy'))
    gt_mask_dict = {}
    for gt_mask in gt_mask_list:
        # 剔除部分数据，tumor114
        if skip_slide(gt_mask):
            continue
        _basename = os.path.basename(gt_mask).split('_resize')[0]
        gt_mask_dict[_basename] = np.load(gt_mask)
    print(f'read GT mask')
    print('Generate train_dataset')
    for i in range(len(otsu_list)):
        otsu = otsu_list[i]
        if skip_slide(otsu):
            continue
        _basename = os.path.basename(otsu).split('_resize')[0]
        add_t,add_n = generate_patch_from_slide(_basename, f_n, f_t, otsu_dict, gt_mask_dict, downsample)
        tumor_count += add_t
        normal_count += add_n
    print(f'Tumor:{tumor_count}\tNormal:{normal_count}')
    f_n.close()
    f_t.close()

if __name__ =='__main__':
    main()
