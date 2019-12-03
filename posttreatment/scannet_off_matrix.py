import os
import torch
import math
from  torch.nn import functional as F
import openslide
from  scannet import Scannet
import PIL
import numpy as np
from  torchvision import transforms
from  torch.autograd import Variable
import time
import glob,os
from  skimage.color import rgb2hsv
from  skimage.filters import threshold_otsu
import pdb


class PostScan():
    def __init__(self, scannet,transform=None,save=None, mini_batch=16,dense_coefficient=2, maxpools=5, stride=2):
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
        self.mini_batch=10


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
#         opt_list.append(opt)
#         opt_list=torch.cat(opt_list,0)
#         print('opt_list size',opt_list.shape)
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
        st = time.time()
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

    def finalprobmap(self,slide_path, roi_path=None, max_k=82,save=None,num_worker=None):
        '''
        max_k = 82按照paper给的结果换算而成，Lp=83，n=83-1=82,用于调控block的大小
        将wsi slide分成多块，然后分开对每个小块求DPT，这里需要提供一个方法解决分块问题，将求得的DPTs缝合起来就得到这张图的最后概率密度图
        Hp代表的是stitched probability map， 或许等于block大小，这里需要测试一下
        设ROI的size为 Lr = 244 + 32k,则 Block的size= ROI+16=260+32k
        如何处理小块？想到的解决方案有：方法1：需要填充，方法2：小块单独计算，此处选择小块单独计算。
        如何提高计算速度：计算量较大
        :param wsi: 输入整张wsi
        :return:final_probability_map 返回最后的概率值
        '''
        st = time.time()
        slide = openslide.open_slide(slide_path)
        basename = os.path.basename(slide_path).rstrip('.tif')
        w, h = slide.dimensions # slide的宽和高
        ki, kj = h//(260+max_k*32),w//(260+max_k*32)  # WSI = sum(BLOCk[i,j],i<=ki,j<=kj)
        print(f'dimension x,y: {w},{h}, block[i,j]:{ki},{kj}')
        fpm_i = ki*(max_k+1)*2 # fpm有i行
        fpm_j = kj*(max_k+1)*2 # fpm有j列
        print(f'fpm_rows:{fpm_i},fpm_columns:{fpm_j}')
        final_probability_map = torch.zeros((fpm_i,fpm_j)).cpu()  # 初始化final_probability_map
        # 添加位置标记
        fpm_column=0 # 标记fpm上columns起始位置
        fpm_row=0 # 标记fpm上rows起始位置
        size= 2*(max_k+1) # dpt的尺寸
        x,y= 0,0 #标记图像slide的起始坐标
        step = 260+max_k*32  # Block的大小,即为在WSI遍历时x,y的步长
        x,y = 0,8700
#         fpm_row,fpm_column = 8700,0
#         block = slide.read_region((x,y),0,(step,step))
#         dpt = self.get_dpt(block,step,step)
#         print("fpm loc: (%d,%d,%d,%d)"%(fpm_row,fpm_row+size,fpm_column,fpm_column+size))
#         final_probability_map[fpm_row:fpm_row+size, fpm_column:fpm_column+size]= dpt
        # 算法效率估计
#         将图像按照max_k的值进行切割，会对剩下的小块进行计算
        for i in range(ki):   # 第i行block,wsi的y
            for j in range(kj): #第j列block,wsi的x
                print(f'block[{i},{j}] size: ({step},{step}),WSI loc:({x},{y})')                   
                block = slide.read_region((x,y),0,(step,step))
                # 拼接pts到fpm中  
                print("fpm _block: (%d,%d,%d,%d)"%(fpm_row,fpm_row+size,fpm_column,fpm_column+size))
                dpt = self.get_dpt(block,step,step)
#                 print(f'fpm size of block[{i},{j}] == dpt size of block: {dpt.shape==(size,size)}')
                final_probability_map[fpm_row:fpm_row+size, fpm_column:fpm_column+size] = dpt
                fpm_column += size # fpm 列平移
                x += step # wsi x平移
                if fpm_column>fpm_j:
                    print(f'{fpm_column}>{fpm_j}')
            fpm_row += size # 移动一个row
            fpm_column =0 #x回到0点
            x = 0 #回到从0开始  
            y += step # y位移

        if self.save:
            npfpm=final_probability_map.numpy()
            filepath=os.path.join(self.save,'%s_fpm.npy'%basename)
            print('savepath:%s'%filepath)
            np.save(filepath,npfpm)
        return final_probability_map
    
    
    def wsi_otsu(self,image):
        """
        input: image(PIL.Image)
        output:
            region_origin - (np.array,m*n*3), 原图数据，用于对比
            region_forward - (np.array,m*n*3), 分割的前景
            region_backward - (np.array,m*n*3), 分割后的背景
            tissue_mask - mask, m*n
            count_true, count_false - otsu阈值保留的有效面积比例
        阈值的来源：是level5全图预先计算的otsu优化值
        默认会占满所有cpu用于加速，同时运行的其他程序会受影响
        """
        region_origin = np.array(image)

        # 颜色空间变换
        img_RGB = np.transpose(region_origin[:, :, 0:3], axes=[1, 0, 2])
        img_HSV = rgb2hsv(img_RGB)
        # otsu阈值处理前背景提取
        background_R = img_RGB[:, :, 0] > 203
        background_G = img_RGB[:, :, 1] > 191
        background_B = img_RGB[:, :, 2] > 201
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > 0.1113
        '''如果仅使用用threadshold，中间会有部份白色脂肪区域被隔离'''
        rgb_min = 50
        min_R = img_RGB[:, :, 0] > rgb_min
        min_G = img_RGB[:, :, 1] > rgb_min
        min_B = img_RGB[:, :, 2] > rgb_min
        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

        count_true = np.sum(tissue_mask == True) # 背景
        count_false = np.sum(tissue_mask == False) # 前景
        return count_true, count_false

    
test_slide_folder = '/root/workspace/dataset/CAMELYON16/testing/images/'
test_slide_annotation_folder = '/root/workspace/dataset/CAMELYON16/testing/lesion_annotations/'

train_slide_folder = '/root/workspace/dataset/CAMELYON16/training/*'
train_slide_annotation_folder = '/root/workspace/dataset/CAMELYON16/training/lesion_annotations/'
slide_list = glob.glob(os.path.join(test_slide_folder, '*.tif'))
slide_list.sort()
print('total slide : %d' % len(slide_list))

pth = '/root/workspace/renqian/0929/save/camelyon16/scannet_train_MSE_NCRF_40w_patch_256/2019-10-21_08-33-34/hardmine_0_epoch_9_type_train_model.pth'
model = Scannet().cuda()
model = torch.nn.DataParallel(model,device_ids=[ 0,1,2,3])
model.load_state_dict(torch.load(pth)['model_state'])
save_npy='/root/workspace/renqian/0929/scannet/11_20/'
if not os.path.exists(save_npy):
    os.mkdir(save_npy)
post = PostScan(scannet=model,save=save_npy)
# 增加断点保存功能
saved=[]
for parent, dirnames, filenames in os.walk(save_npy):
    for filename in filenames:
        saved.append(filename.rstrip('_fpm.npy'))
print('saved:',saved)  
for slide_path in slide_list: 
    filename=os.path.basename(slide_path).rstrip('.tif')
#     if filename == 'test_002':
#         final_probability_map=post.finalprobmap(slide_path,max_k=10)
    print(filename in saved)
    if filename in saved:
        continue
    print(slide_path)
    sttime=time.time()
    final_probability_map=post.finalprobmap(slide_path,max_k=50)
    end=time.time()
    print('total time %f'%(end-sttime))