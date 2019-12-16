import sys,os
sys.path.append('..')
import torch
import math
from  torch.nn import functional as F
import openslide
from  basic.model.scannet import Scannet
import PIL
import numpy as np
from  torchvision import transforms
from  torch.autograd import Variable
import time
import glob,os
from  skimage.color import rgb2hsv
from  skimage.filters import threshold_otsu



class PostScan():
    def __init__(self, scannet,transform=None,save=None,dense_coefficient=2, maxpools=5, stride=2):
        '''

        :param scannet: scannet 模型
        :param dpt_size: 输出dpt的大小
        :param dense_coefficient: DPTs/OPTs的尺寸比例
        :param maxpools: 最大池化层数
        :param stride: 模型步长
        :param save: 数据保存路径 
        '''
        self.model = scannet
        self.alpha = int(dense_coefficient) # ratio between the size of DPTs and the size of OPT
        self.sf = int(math.pow(stride, maxpools))  # 求出的Sf是FCN的滑动步长 inner stride of scannet
        self.sd = int(self.sf / self.alpha)  # 偏移量Sd
        self.lf = 244  # 输入Scannet的概率图
        self.transform = transform
        self.save =save



    def getopt(self,opts,roi_list):
        '''(测试通过)
        计算Block区域内部多个ROI的概率矩阵
        设定的roi是PIL.Image类
        Lr = Lf + (Lp -1) * Sf; Sr = Sf *Lp
        假设Lr = 2868，Sf=32，Lf=244，则Lp=83(吻合，此处ok),此时opt大小为LpXLpX2,经过softmax转换成LpXLpX1的p值
        :param roi: 单个ROI区域
        :return: opt矩阵
        '''
        
        roi_batch=torch.cat(roi_list,0)
        opt = self.model(roi_batch)
        opt =F.softmax(opt)[:,1].cpu().detach()
        count=0
        for i in range(self.alpha):
            for j in range(self.alpha):
                opts[i,j,:,:]=opt[count,]
                count +=1

    def get_dpt(self, block,wi,hi):
        '''(测试通过)
        给定一个dpt大小的图像，生成对应的dpt
        设image， PIL.Image类
        假设Lr= 2868，Sf=32, Sd=Sf/alpha=32/2=16，Lf=244; 
        block大小应该为2868+（alpha-1）*16 = 2884 对应的opt = Lp * Lp * alpha * alpha
        由alpha的定义可知 len_dpt = alpha * len_opt =wei_dpt = alpha * len_opt =alpha * Lp
        :param block: 输入dpt对应的图像block
        :param hi,wi:输入block的尺寸
        :return:dpt (测试已通过)
        '''
        def interweaving(dpt,opts):
            '''
            After scan image, we can reconstuct DPT by inter.
            :return:
            '''
            W, H = dpt.shape
            for h_ in range(H):
                for w_ in range(W):
                    i = h_ % self.alpha
                    j = w_ % self.alpha
                    h = int(h_ / self.alpha)
                    w = int(w_ / self.alpha)
                    dpt[w_, h_] = opts[i, j][w, h]
            return dpt
        x, y = 0, 0
        # st = time.time()
        hp, wp = int(hi - self.sd *(self.alpha - 1)), int(wi - self.sd * (self.alpha - 1))  #计算ROI区域大小
        ho, wo = int((hp - self.lf) / self.sf) + 1, int((wp - self.lf) / self.sf) + 1 #计算ROI区域的Lp值
        opts = torch.zeros((self.alpha,self.alpha,wo,ho)).cpu() # 初始化opts矩阵
        dpt = torch.zeros((self.alpha*wo,self.alpha*ho)).cpu()  # 初始化dpts矩阵
#         print('dpt.shape')
#         print(dpt.shape)
        roi_list=[]
    # 将 roi打包成batch_size
        for i in range(self.alpha):
            for j in range(self.alpha):
                roi = block.crop((x, y, x+wp, y+hp)).convert('RGB')  # left, upper, right, lower
                roi=transforms.ToTensor()(roi).permute(0,2,1).unsqueeze(dim=0)
                roi_list.append(roi)
                y +=  self.sd
            x +=  self.sd
        # 计算batch_size的pValue
        self.getopt(opts,roi_list)
        time1 = time.time()
#         print('opts time:',time1-st)
        dpt = interweaving(dpt,opts)
#         print('dpts:',time.time()-time1)
        return dpt


    def densereconstruction(self,slide_path,otsu,resize,max_k=82,threshold=0.1):
        '''
        :param slide_path:
        :param roi_path:
        :param max_k:
        :return: dense #最终概率密度
        '''
        slide = openslide.open_slide(slide_path)
        basename = os.path.basename(slide_path).rstrip('.tif')
        w, h = slide.dimensions  # slide的宽和高
        dense_i = h//self.sd
        dense_j = w//self.sd
        dense = torch.zeros((dense_i, dense_j)).cpu()  # 初始化dense
        size = self.alpha*(max_k+1)
        k_i = dense_i//size # 分成多块 行
        k_j = dense_j//size # 分成多块 列
        step = 260 + max_k * 32 # 每个WSI上区域的大小
        size = otsu.size
        def filterregion(i_st, j_st,size):
            gap = step//resize
            count = np.sum(otsu[i_st,j_st])
            return count//size < threshold

        for i in range(k_i):
            for j  in range(k_j):
                x,y=j*size*self.sd-122,  i*size*self.sd-122 # WSI 上的起始坐标从-122开始
                # 映射到otsu的坐标
                i_st,j_st = y//resize,x//resize
                if filterregion(i_st,j_st,size):
                    continue
                block = slide.read_region((x, y), 0, (step, step))
                dpt = self.get_dpt(block, step, step)
                dense[i*size:(i+1)*size,j*size:(j+1)*size]=self.get_dpt(block,step,step)
        if self.save:
            npfpm = final_probability_map.numpy()
            filepath = os.path.join(self.save, '%s_fpm.npy' % basename)
            print('savepath:%s' % filepath)
            np.save(filepath, npfpm)
        return dense


test_slide_folder = '/root/workspace/dataset/CAMELYON16/testing/images/'
test_slide_annotation_folder = '/root/workspace/dataset/CAMELYON16/testing/lesion_annotations/'
resize = 32
test_slide_ostu = '~/workspace/huangxs/prepare_data/16/wsi_otsu_save/test_resize_%d'%resize

train_slide_folder = '/root/workspace/dataset/CAMELYON16/training/*'
train_slide_annotation_folder = '/root/workspace/dataset/CAMELYON16/training/lesion_annotations/'
slide_list = glob.glob(os.path.join(test_slide_folder, '*.tif'))
slide_list.sort()
print('total slide : %d' % len(slide_list))
pth = '/root/workspace/renqian/0929/save/camelyon16/scannet_train_MSE_NCRF_40w_patch_256/2019-10-21_08-33-34/hardmine_0_epoch_9_type_train_model.pth'

model = Scannet().cuda()
model.eval()
model = torch.nn.DataParallel(model,device_ids=[ 0,1,2,3])
model.load_state_dict(torch.load(pth)['model_state'])
save_npy='/root/workspace/renqian/0929/scannet/11_20/'
with open(os.path.join(save_npy,'log.txt'),'w')as f:
    f.write(pth+'\n'+save_npy)
if not os.path.exists(save_npy):
    os.mkdir(save_npy)
post = PostScan(scannet=model,save=save_npy,dense_coefficient=1)
# 增加断点保存功能
saved=[]
for parent, dirnames, filenames in os.walk(save_npy):
    for filename in filenames:
        saved.append(filename.rstrip('_fpm.npy'))
print('saved:',saved)  
for slide_path in slide_list: 
    filename=os.path.basename(slide_path).rstrip('.tif')
    ostu = np.load(os.path.join(test_slide_ostu,filename+'_resize_%d.npy'%resize))
    final_probability_map=post.densereconstruction(slide_path,otsu,resize,omax_k=10)