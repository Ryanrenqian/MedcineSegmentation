#! /usr/bin/python
import argparse, os, pdb, glob, json
from multiprocessing import Pool
import numpy as np
import math
import matplotlib.pyplot as plt
import logging


def statistics(slide_label, slide_avg_prob):
    """
    calculate FPR TPR Precision Recall IoU
    :return: (FPR, TPR, AUC),(Precision,Recall,MAP),IoU
    para:
        slide_label, N dim dict {xxx.tif:label=0,1 }
        slide_avg_prob, N dim list {xxx.tif, prob[0.0, 1.0]}
    """
    TP = []
    FP = []
    FN = []
    TN = []
    for threshold in np.arange(0, 1.1, 0.001):
        temp_TP = 0.0
        temp_FP = 0.0
        temp_FN = 0.0
        temp_TN = 0.0
        #         assert (len(slide_label) == len(slide_avg_prob))

        for k in slide_avg_prob.keys():
            _name = k
            if '114' in k:  # 114需要排除出计算范围
                continue
            _prob = slide_avg_prob[_name]
            _label = slide_label[_name]
            _prob_label = 1 if (_prob >= threshold) else 0
            _prob_correct = _label == _prob_label

            if _label == 1:  # 正样本
                temp_TP += 1 if _prob_correct else 0
                temp_FN += 0 if _prob_correct else 1  #

            if _label == 0:  # 负样本
                temp_FP += 0 if _prob_correct else 1
                temp_TN += 1 if _prob_correct else 0
        TP.append(temp_TP)
        FP.append(temp_FP)
        FN.append(temp_FN)
        TN.append(temp_TN)
    TP = np.asarray(TP).astype('float32')
    FN = np.asarray(FN).astype('float32')
    TN = np.asarray(TN).astype('float32')
    FP = np.asarray(FP).astype('float32')
    TPR = (TP) / (TP + FN + 0.0001)  # 防止为0
    Specificity = (TN) / (TN + FP + 0.0001)  # 防止为0
    FPR = 1 - Specificity
    Precision = (TP) / (TP + FP + 0.0001)  # 防止为0
    Recall = TPR
    AUC = np.round(np.sum((TPR[1:] + TPR[:-1]) * (FPR[:-1] - FPR[1:])) / 2., 4)
    return TP, FP, TN, FN, TPR, FPR, Precision, Recall, AUC


# 画ROC曲线及AUC
def draw_auc(_FPR, _TPR, _AUC, save_dir):
    font_size = 16
    plt.figure(figsize=(9, 9))
    plt.rcParams['savefig.dpi'] = 72  # 图片像素
    plt.rcParams['figure.dpi'] = 72
    metirc = np.arange(0, 1.01, 0.01)
    plt.xlabel("FPR(false positive rate)", fontsize=font_size)
    plt.ylabel("TPR(true positive rate)", fontsize=font_size)
    plt.plot(metirc, metirc, 'g--', linewidth=2, label=('%d' % 1))
    plt.plot(_FPR, _TPR, 'r-', linewidth=2, label=('%d' % 1))
    plt.text(0.8, 0.2, 'AUC=%.4f' % _AUC, fontsize=font_size)
    plt.title('AUC PLOT')
    filename = os.path.join(save_dir, 'auc.png')
    plt.savefig(filename)


# TP,FP,TN,FN in
def patch_stats(slide_result):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in slide_result:
        if i["gt_label"] == 1:
            if i["is_correct"] == True:
                TP += 1
            else:
                FN += 1
        elif i["gt_label"] == 0:
            if i["is_correct"] == True:
                TN += 1
            else:
                FP += 1
    return TP, FP, TN, FN


# read pred_list and sort it
def read_pred(pred_list):
    slide_result_txt_list = glob.glob(os.path.join(pred_list, '*.tif.txt'))
    slide_result_txt_list.sort()
    slide_result_dict = {}
    slide_prob_dict = {}
    TP_list, FP_list, TN_list, FN_list = [], [], [], []
    for slide_result_txt_path in slide_result_txt_list:
        slide_basename = slide_result_txt_path.split('slide_')[1].split('.txt')[0]
        with open(slide_result_txt_path, 'r')as f:
            content = f.read()
        slide_result = json.loads(content)
        slide_result.sort(key=lambda x: x['pred_score'], reverse=True)
        slide_result_dict[slide_basename] = slide_result
        slide_prob_dict[slide_basename] = [i['pred_score'] for i in slide_result]
        TP, FP, TN, FN = patch_stats(slide_result)
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
    TP, FP, TN, FN = np.sum(TP_list), np.sum(FP_list), np.sum(TN_list), np.sum(FN_list)
    logging.info(f'TP:{TP} , FP: {FP}, TN:{TN}, FN:{FN}')
    logging.info(
        f'precision: {TP / (TP + FP)},Recall: {TP / (TP + FN)},Accuracy: {(TP + TN) / (TP + FP + TN + FN)},Specificity:{(TN) / (TN + FP)}')
    return slide_result_dict, slide_prob_dict


# read mask and generate all prob map to npfile
def mask_prop(mask_folder, slide_result_dict, prob_map_folder):
    mask_list = glob.glob(os.path.join(mask_folder, '*.npy'))
    mask_list.sort()
    prob_map_dict = {}
    for mask in mask_list:
        basename = os.path.basename(mask.split('_resize_')[0]) + '.tif'
        mask_np = np.load(mask)
        prob_map_dict[basename] = np.zeros(mask_np.shape)
        prob_map = prob_map_dict[basename]
        if basename in slide_result_dict.keys():
            slide_result_list = slide_result_dict[basename]
            for result in slide_result_list:
                x, y = result['path'].split('_')[-2:]
                x, y = int(int(x) / 64), int(int(y) / 64)
                prob = result['pred_score']
                prob_map[x, y] = prob
        # save prob map to *.npy
        prob_map_npy = os.path.join(prob_map_folder, basename.split('.tif')[0])
        np.save(prob_map_npy, prob_map)


# 可视化prob map生成图像
def save_prob_map(prob_map, save_folder, basename):
    pseudo_np = np.zeros([prob_map.shape[0], prob_map.shape[1], 3])
    pseudo_np[:, :, 0] = prob_map * 255
    pseudo_np[:, :, 1] = prob_map * 255
    pseudo_np[:, :, 2] = prob_map * 255 * 0.5
    prob_visual = Image.fromarray(np.uint8(np.transpose(pseudo_np, (1, 0, 2))))
    prob_visual.save(os.path.join(save_folder, 'prob_map_slide_%s.png' % basename))


def generate_csv_by_nms(probs_map, prob_thred, csv_path, radius=12):
    """
    用NMS的方式获取分割的结果
    """
    wsi_max_prob = 0
    X, Y = probs_map.shape
    resolution = 64
    outfile = open(csv_path, 'w')

    while np.max(probs_map) > prob_thred:
        prob_max = probs_map.max()
        max_idx = np.where(probs_map == prob_max)
        x_mask, y_mask = max_idx[0][0], max_idx[1][0]
        x_wsi = int((x_mask + 0.5) * resolution)
        y_wsi = int((y_mask + 0.5) * resolution)
        outfile.write('{:0.5f},{},{}'.format(prob_max, x_wsi, y_wsi) + '\n')
        x_min = (x_mask - radius) if (x_mask - radius) > 0 else 0
        x_max = (x_mask + radius) if (x_mask + radius) <= X else X
        y_min = (y_mask - radius) if (y_mask - radius) > 0 else 0
        y_max = (y_mask + radius) if (y_mask + radius) <= Y else Y
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                probs_map[x, y] = 0
        # 如果周围的均值都属于tumor，则视为tumor
    #         if avg:
    #             if probs_map[x_min:x_max, y_min:y_max].mean() > prob_thred*0.25:
    #                 outfile.write('{:0.5f},{},{}'.format(prob_max, x_wsi, y_wsi) + '\n')
    #                 wsi_max_prob = max(wsi_max_prob, prob_max)
    #                 wsi_max_prob = max(wsi_max_prob, probs_map[x_min:x_max, y_min:y_max].mean())
    #         else:
    #             outfile.write('{:0.5f},{},{}'.format(prob_max, x_wsi, y_wsi) + '\n')
    #             wsi_max_prob = max(wsi_max_prob, prob_max)
    #         probs_map[x_min:x_max, y_min:y_max] = 0
    outfile.close()



def getargs():
    parser = argparse.ArgumentParser(description='cacluate roc and auc')
    parser.add_argument('-sf', '--slide_folder', type=str, default='/root/workspace/dataset/CAMELYON16/testing/images/',
                        metavar='DIR', help='slide_folder')
    parser.add_argument('-saf', '--slide_annotation_folder', type=str,
                        default='/root/workspace/dataset/CAMELYON16/testing/lesion_annotations/', metavar='DIR',
                        help='slide_annotation_folder')
    parser.add_argument('-mf', '--mask_folder', type=str,
                        default='/root/workspace/huangxs/prepare_data/16/wsi_mask/test_64/', help='mask_folder')
    parser.add_argument('-t', '--thred', default=0.5, type=float, help='threshhold')
    parser.add_argument('-o', '--output',  type=str,help='output')
    parser.add_argument('-i', '--input', type=str,help="csv folder")
    return parser.parse_args()

def main():

    args = getargs()
    os.system(f'mkdir -p {args.output}')
    logfile = os.path.join(args.output, 'log.txt')
    logging.basicConfig(level=logging.INFO, filename=logfile,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(args)
    logging.basicConfig(level=logging.INFO, filename=logfile,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    csv_folder = os.path.join(args.output, 'csv')
    slide_prob_nms = {}
    csv_list = glob.glob(os.path.join(csv_folder, '*.csv'))
    slide_result={}
    for csv_path in csv_list:
        f = open(csv_path, 'r')
        lines = f.readlines()
        f.close()
        tif_name = os.path.basename(csv_path).split('.')[0] + '.tif'
        prob_max = 0
        for line in lines:
            prob, x, y = line.split(',')
            prob_max = max(prob_max, float(prob))
        slide_prob_nms[tif_name] = float(prob_max)
    logging.info(f'example: slide_prob_nms[{tif_name}]:{slide_prob_nms[tif_name]}')
    #     pdb.set_trace()
    slide_gt_dict = {}
    slide_list = glob.glob(os.path.join(args.slide_folder, '*.tif'))
    for slide_name in slide_list:
        slide_basename = os.path.basename(slide_name)
        slide_gt_dict[slide_basename] = 0
        prefix = slide_basename.split('.')[0]
        if len(glob.glob('%s%s*' % (args.slide_annotation_folder, prefix))):
            slide_gt_dict[slide_basename] = 1
        slide_result[slide_basename]=(slide_prob_nms[slide_basename],slide_gt_dict[slide_basename])
    TP, FP, TN, FN, TPR, FPR, Precision, Recall, AUC = statistics(slide_gt_dict, slide_prob_nms)
    result_path=os.path.join(args.output,'result.json')
    with open(result_path, 'w') as f:
        json.dump(slide_result, f,indent=4)
    logging.info(f'AUC:{AUC},TP:{TP},FP:{FP},TN:{TN},FN:{FN}')
    save_dir = os.path.join(args.output, 'figures')
    os.system(f'mkdir -p {save_dir}')
    draw_auc(FPR, TPR, AUC, save_dir)


if __name__ == '__main__':
    main()