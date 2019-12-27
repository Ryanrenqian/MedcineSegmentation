import sys, os,logging,glob,time
sys.path.append('..')
from basic.posttreatment import ScanNetPost,Scan
from basic.model import Scannet

import torch
import numpy as np
import json
import multiprocessing
from skimage import measure, color, morphology

def DenseStruct(config,slide_list,dense):
    workspace=config["workspace"]
    os.system(f'mkdir -p {workspace}')

    k = config["k"]
    thres=config["thres"]
    pth = config["pth"]

    resize = config["resize"]
    slide_ostu = os.path.join(config["otsu"], 'test_resize_%d' % resize)
    logfile = os.path.join(workspace, 'log.txt')
    #logging.basicConfig(level=logging.INFO, filename=logfile,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    save_npy = os.path.join(workspace,f'dense%{dense}')

    os.system(f'mkdir -p {save_npy}')
    model = Scannet().cuda()
    model.eval()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(pth)['model_state'])

    logging.info('total slide : %d' % len(slide_list))
    with open(os.path.join(save_npy, 'log.txt'), 'w')as f:
        f.write(pth + '\n' + save_npy)
        f.write(str(config))
    post = Scan(scannet=model, save=save_npy, dense_coefficient=dense)
    # 增加断点保存功能
    saved = []
    for parent, dirnames, filenames in os.walk(save_npy):
        for filename in filenames:
            saved.append(filename.rstrip('_fpm.npy'))
    logging.info('saved:', saved)
    for slide_path in slide_list:
        filename = os.path.basename(slide_path).rstrip('.tif')
        st = time.time()
        if filename in saved:
            logging.info(f'pass {filename}')
            continue
        otsu = np.load(os.path.join(slide_ostu, filename + '_resize_%d.npy' % resize))
        logging.info(f"handle {filename}: {np.sum(otsu)} ostu forward")
        post.densereconstruction(slide_path, otsu, resize, max_k=k, threshold=thres)
        ed = time.time()
        logging.info(f'time: {ed - st} in {filename}')

def ProcessDense(process,sample_list,workspace,dense,kernel,interval=3600):
    '''
    Parameters
    ----------
    process
    sample_list
    workspace
    dense
    kernel

    Returns
    -------

    '''
    # set logging
    os.system(f'mkdir -p {workspace}')
    kernelpath=os.path.join(workspace,f'kernel_{kernel}')
    logfile =  os.path.join(kernelpath, '%d.log' % kernel)
    FORMAT = '%(asctime)-15s-8s %(message)s'
    logging.basicConfig(filename=logfile, format=FORMAT, level=logging.INFO)
    fpm_folder = os.path.join(workspace, f'dense{dense}')
    save_folder = os.path.join(workspace, 'kernel_%d/csv' % kernel)
    #     processing Pool
     # 设置进程池大小为40
    # begin
    os.system('mkdir -p %s' % save_folder)
    fpm_list = glob.glob(os.path.join(fpm_folder, '*.npy'))
    saved = []
    logging.info("start process dense")
    while(len(saved)!=len(sample_list) ):
        for fpm_name in fpm_list:
            sample = os.path.basename(fpm_name).rstrip('_fpm.npy')
            save_path = os.path.join(save_folder, sample)
            if os.path.exists(save_path + '.csv'):
                saved.append(sample)
                continue
            fpm = np.load(fpm_name)
            process.fix(fpm,save_path,True)
            time.sleep(interval)  #设置扫描间隔
    logging.info('Process end.')


def main():
    # Set config
    configpath=sys.argv[1]
    with open(configpath, 'r')as f:
        config = json.load(f)
    workspace=config["workspace"]
    os.system(f'mkdir -p {workspace}')
    slide_folder = config["slide_folder"]
    slide_list = glob.glob(os.path.join(slide_folder, '*.tif'))
    slide_list.sort()
    dense = config['dense']
    kernel = config['kernel']
    logging.info("start")
    # ProcessDense
    p = multiprocessing.Pool(config["pools"])
    for i in range(1, kernel):
        process=ScanNetPost(sd=32 / dense, kernel=morphology.square(i))
        p.apply_async(ProcessDense,args=[process,slide_list,workspace,dense])
    p.close()
    p.join()
    # Generate Dense
    # DenseStruct(config,slide_list,dense)

if __name__ == "__main__":
    main()