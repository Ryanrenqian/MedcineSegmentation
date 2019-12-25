import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import os
import sys
import cv2
import json,argparse,logging
import math
from skimage.measure import points_in_poly
EVALUATION_MASK_LEVEL = 0
L0_RESOLUTION = 0.243

def readCSVContent(csvDIR):
    """Reads the data inside CSV file
    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image

    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    Xcorr, Ycorr, Probs = ([] for i in range(3))
    csv_lines = open(csvDIR, "r").readlines()
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        Probs.append(float(elems[0]))
        Xcorr.append(int(elems[1]))
        Ycorr.append(int(elems[2]))
    return Probs, Xcorr, Ycorr


def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve

    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image

    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """

    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    # logging.info(len(all_probs[1:]))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs) / float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs) / float(sum(FROC_data[3])+0.0000001)
    return total_FPs, total_sensitivity


def plotFROC(total_FPs, total_sensitivity,save=None):
    """Plots the FROC curve

    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds

    Returns:
        -
    """
    fig = plt.figure()
    plt.figure(figsize=(9, 9))
    plt.rcParams['savefig.dpi'] = 72  # 图片像素
    plt.rcParams['figure.dpi'] = 72
    plt.xlabel('Average Number of False Positives', fontsize=18)
    plt.ylabel('Metastasis detection sensitivity', fontsize=18)
    fig.suptitle('Free response receiver operating characteristic curve', fontsize=18)
    plt.plot(total_FPs, total_sensitivity, '-', color='#000000')
    filename = os.path.join(save, 'froc.png')
    plt.savefig(filename)


def judgeinpoly(x,y,coord):
    """
    Judge whether (x,y) is in coord
    """
    HittedLabel = 0
    for annotation in coord['positive']:
        name = annotation['name']
        vertices = np.array(annotation['vertices'])
        In_poly = points_in_poly([(x,y)],vertices)
        #In_poly = points_in_poly([(y,x)],vertices)
        if In_poly:
            temp = name[11:]
            HittedLabel = int(temp) + 1
            return HittedLabel
    return HittedLabel


def computeITC(mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)

    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object
        should be less than 275µm to be considered as ITC (Each pixel is
        0.243µm*0.243µm in level 0). Therefore the major axis of the object
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.

    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made

    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = len(mask['positive'])
    Isolated_Tumor_Cells = []
    threshold = 275 / (resolution * pow(2, level))
    for annotation in mask['positive']:
        name = annotation['name']
        vertices = np.array(annotation['vertices'])
        rect = cv2.minAreaRect(vertices)
        x, y = rect[1]
        length = math.sqrt(pow(x, 2) + pow(y, 2))
        if length < threshold:
            temp = name[11:]
            i = int(temp)
            Isolated_Tumor_Cells.append(i + 1)
    return Isolated_Tumor_Cells

def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, coord, ITC_labels):
    max_label = len(coord['positive'])
    FP_counter = 0
    max_label = len(coord['positive'])
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    logging.info('ITC_labels:'+str(ITC_labels))
    if (is_tumor):
        for i in range(0,len(Xcorr)):
            HittedLabel = judgeinpoly(Xcorr[i],Ycorr[i],coord)
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                FP_counter+=1
            elif HittedLabel not in ITC_labels:
                if (Probs[i]>TP_probs[HittedLabel-1]):
                    TP_probs[HittedLabel-1] = Probs[i]
    else:
        for i in range(0,len(Xcorr)):
            FP_probs.append(Probs[i])
            FP_counter+=1
    num_of_tumors = max_label-len(ITC_labels);
    return FP_probs, TP_probs, num_of_tumors

def getargs():
    parser = argparse.ArgumentParser(description='calculate FROC')
    parser.add_argument('-m', '--mask_folder', type=str,
                        default='/root/workspace/huangxs/prepare_data/16/wsi_mask/test_64/', help='mask_folder')
    parser.add_argument('-o', '--output',  type=str,help='output')
    parser.add_argument('-i', '--input', type=str,help="csv folder")
    return parser.parse_args()

def main():
    args=getargs()
    logfile = os.path.join(args.output, 'log.txt')
    logging.basicConfig(level=logging.INFO, filename=logfile,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(args)
    mask_folder=args.mask_folder
    result_file_list = []
    result_folder=args.input
    for i in range(1, 131):
        if i == 114:
            continue
        if i < 10:
            test = 'test_00'
        elif i < 100:
            test = 'test_0'
        elif i > 100:
            test = 'test_'
        dir = test + str(i) + '.csv'
        if os.path.exists(result_folder + dir):
            result_file_list.append(dir)
    FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    ground_truth_test = []
    ground_truth_test += [each[0:8] for each in os.listdir(mask_folder) if each.endswith('.json')]
    ground_truth_test = set(ground_truth_test)
    caseNum = 0
    for case in result_file_list:
        logging.info(f'Evaluating Performance on image:{case[0:-4]}')
        sys.stdout.flush()
        csvDIR = os.path.join(result_folder, case)
        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)
        is_tumor = case[0:-4] in ground_truth_test
        logging.info(is_tumor)
        if (is_tumor):
            maskDIR = os.path.join(mask_folder, case[0:-4]) + '.json'  # .tif
            with open(maskDIR, 'r') as load_f:
                mask = json.load(load_f)
            ITC_labels = computeITC(mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
        else:
            mask = {'positive': []}
            ITC_labels = []

        FROC_data[0][caseNum] = case
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case
        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs,
                                                                                                  is_tumor, mask,
                                                                                                  ITC_labels)
        caseNum += 1
    total_FPs, total_sensitivity = computeFROC(FROC_data)
    total_FPs.sort()
    total_sensitivity.sort()
    points = len(total_FPs[total_FPs < 800])
    plotFROC(total_FPs[:points], total_sensitivity[:points],args.output)
    # compute avg froc
    eval_threshold = [0.25, 0.5, 1, 2, 4, 8]
    eval_TPs = np.interp(eval_threshold, total_FPs[:], total_sensitivity[:])
    for i in range(len(eval_threshold)):
        logging.info('Avg FP = %.2f  Sensitivity = %.4f' % (eval_threshold[i], eval_TPs[i]))
    logging.info('\navg score @FP %s = %.4f' % (str(eval_threshold), np.mean(eval_TPs)))
    # compute avg froc
    eval_threshold = [0.25, 0.5, 1, 2, 4, 8000]
    eval_TPs = np.interp(eval_threshold, total_FPs[:], total_sensitivity[:])
    for i in range(len(eval_threshold)):
        logging.info('Avg FP = %.2f  Sensitivity = %.4f' % (eval_threshold[i], eval_TPs[i]))
    logging.info('\navg score @FP %s = %.4f' % (str(eval_threshold), np.mean(eval_TPs)))

if __name__=='__main__':
    main()