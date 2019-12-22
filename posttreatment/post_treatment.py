import numpy as np
import sys, pdb
sys.path.append('..')
import glob, os
import matplotlib.pyplot as plt
import csv
from skimage import measure, color, morphology
import numpy as np
import matplotlib.pyplot as plt
import threading
import multiprocessing
import time, logging
import logging
import argparse

class ScanNetPost():
    def __init__(self, sd=16, lf=244, connectivity=2, kernel=np.ones((10, 10)), threshhold=0.5):
        '''
        初始化后处理的相关参数
        :param kernel_size:
        :param connectivity:
        '''
        self.kernel = kernel
        self.connectivity = connectivity
        self.threshhold = threshhold
        # 卷积
        #         self.square_dim
        self.Sd = sd
        self.Lf = lf

    def fix(self, fpm, save_path=None, img=False, show=False):
        '''
        进行后处理的操作
        :param fpm:
        :return:
        '''

        logging.info('Handling %s' % save_path)
        st = time.time()
        bifpm = np.where(fpm > self.threshhold, 1, 0)  # 二分类
        time1 = time.time()
        logging.info('calculate bifpm time consuming: %.4f' % (time1 - st))
        fpm = np.where(fpm > self.threshhold, fpm, 0)
        opening = morphology.opening(bifpm, self.kernel)
        time2 = time.time()
        logging.info('calculate opening time consuming: %.4f' % (time2 - time1))
        labels, num_label = measure.label(opening, connectivity=self.connectivity, return_num=True)
        logging.info('num_label = %d' % num_label)
        fpm_filter = fpm * opening
        csvRows = []  # 保存中心点和概率值
        for i in range(num_label):
            #             pdb.set_trace()
            st = time.time()
            max_p = np.max(fpm_filter[labels == i])
            ed = time.time()
            logging.info('each Iter, max_p st-ed: %.4f' % (ed - st))
            fpm_filter[labels == i] = max_p  # 取最大概率
            ed2 = time.time()
            logging.info('each Iter, filter st-ed: %.4f' % (ed2 - ed))
            indexes = np.argwhere(labels == i)
            x, y = np.sum(indexes, axis=0) / indexes.shape[0]
            x = int(x * self.Sd + self.Lf / 2)  # 使用中间值map
            y = int(y * self.Sd + self.Lf / 2)  # 使用中间值map
            csvRows.append([max_p, y, x])  # Transpose location
            ed3 = time.time()
            logging.info('each Iter, index st-ed: %.4f' % (ed3 - ed2))

        time3 = time.time()
        logging.info('recreate fpm time consuming: %.4f' % (time3 - time2))
        csvRowSorted = sorted(csvRows, key=lambda x: x[0], reverse=True)  # 根据概率排序
        csvArray = np.array(csvRows)
        if img:
            plt.rcParams['figure.dpi'] = 300
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(fpm, cmap=plt.cm.gray)
            ax1.set_title('Origin')
            ax2.imshow(opening, cmap=plt.cm.gray)
            ax2.set_title('open operation')
            ax3.imshow(fpm_filter, cmap=plt.cm.gray)
            ax3.set_title('contours maximum')
            if show:
                plt.show()
            elif save_path:
                img = os.path.join(save_path)
                plt.savefig(img)
        if save_path:
            csvname = save_path + '.csv'
            try:
                np.savetxt(csvname, csvArray, fmt=['%.4f', '%d', '%d'], delimiter=",")
            except AttributeError:
                logging.info('fpm is empty, touch a empty file: %s' % csvname)
                os.system('touch %s' % csvname)
        logging.info('%s Done' % save_path)


#         return fpm_filter  # 无需返回值


def multprocess(process, fpm, save_path, share_dict, lock):
    process.fix(fpm, save_path, img=True)
    lock.acquire()
    share_dict['running_process'] -= 1
    lock.release()

def getargs():
    parser = argparse.ArgumentParser(description='dense post treatment')
    parser.add_argument('-k','--kernel_size',default=8,type=int)
    parser.add_argument('-p', '--pool_size', default=8, type=int)
    parser.add_argument('-s', '--save',  type=str,help='workspace')
    parser.add_argument('-i', '--input', type=str,help="fpm folder")
    return parser.parse_args()

if __name__ == '__main__':
    args=getargs()
    kernel = args.kernel_size  # kernel size
    pools = args.pool_size  # processing pool size
    # set logging
    workspace = args.save
    os.system(f'mkdir -p {workspace}')
    logfile = workspace + 'kernel_%d.log' % kernel
    FORMAT = '%(asctime)-15s-8s %(message)s'
    logging.basicConfig(filename=logfile, format=FORMAT, level=logging.INFO)
    fpm_folder = args.input
    save_folder = os.path.join(workspace, 'kernel_%d/csv' % kernel)
    #     processing Pool
    p = multiprocessing.Pool(pools)  # 设置进程池大小为40
    # begin
    os.system('mkdir -p %s' % save_folder)
    fpm_list = glob.glob(os.path.join(fpm_folder, '*.npy'))
    process = ScanNetPost(kernel=morphology.square(kernel))
    for fpm_name in fpm_list:
        sample = os.path.basename(fpm_name).rstrip('_fpm.npy')
        save_path = os.path.join(save_folder, sample)
        if os.path.exists(save_path + '.csv'):
            continue
        fpm = np.load(fpm_name)
        p.apply_async(func=process.fix, args=(fpm, save_path, True, False))
    #         process.fix(fpm,save_path,True)
    p.close()
    p.join()
    logging.info('Process end.')