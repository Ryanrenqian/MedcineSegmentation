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
#         print('roi batch size',roi_batch.shape)
        sample_size=roi_batch.shape[0]
        Iteration = int(sample_size/self.mini_batch)
        opt_list=[]
        rows=0
        while(rows*self.mini_batch+self.mini_batch<sample_size):
            mini_batch=roi_batch[rows*self.mini_batch:(rows+1)*self.mini_batch]
            opt = self.model(mini_batch)
            opt =F.softmax(opt)[:,1].cpu().detach()
            opt_list.append(opt)
            rows+=1
        mini_batch=roi_batch[rows*self.mini_batch:sample_size]
        opt = self.model(mini_batch)
        opt =F.softmax(opt)[:,1].cpu().detach()
        opt_list.append(opt)
        opt_list=torch.cat(opt_list,0)
        print('opt_list size',opt_list.shape)
        count=0
        for i in range(self.alpha):
            for j in range(self.alpha):
                opts[i,j,:,:]=opt_list[count]
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
        w, h = slide.dimensions
        print(w,h)
        kh, kw = int(h/(260+max_k*32)),int(w/(260+max_k*32))  #求w,h对应的k值，既能够划分为max_k的block的块数，会有剩余的小块
        lh, lw = h%(260+max_k*32), w%(260+max_k*32) #求小块的大小
        lkh,lkw=int((lh-260)/32), int((lw-260)/32) # 求小块的k值
        print(lkh,lkw)
        skip_w = 1 # 是否要计算w的最后一块,默认为计算
        skip_h = 1 # 是否要计算h的最后一块,默认为计算
        # 初始化final_probability_map的大小
        if lw-260<0:  # 如果剩余块大小小于260则去掉该块(无法计算)
            x = 2*(kw * (max_k+1))   # final_probability_map的维度x
            skip_w = 0
        else:
            x = 2*(kw * (max_k+1) + lkw+1)  # final_probability_map的维度x
        if lh-260<0:  # 如果剩余块大小小于260则去掉该块(无法计算)
            y = 2*(kh * (max_k+1)) #  final_probability_map的维度y
            skip_h = 0
        else:
            y = 2*(kh * (max_k+1) + lkh+1)  # final_probability_map的维度y
#         print(skip_h,skip_w)
        print('time prediction:',(x),'*',(y))
#         time1=time.time()
#         print('Init fpm',time1-st)
        print('fpm size: (%d,%d)'%(x,y))
        final_probability_map = torch.zeros((x,y)).cpu()  # 初始化final_probability_map
        # 添加位置标记
        flag_x=0 # 标记fpm x起始位置
        flag_x_w=0 # 标记x方向的长度
        flag_y=0 # 标记fpm y起始位置
        flag_y_h=0 # 标记fpm y方向长度
        h,w= 0,0 #标记图像y的起始坐标
        # 算法效率估计
        # 将图像按照max_k的值进行切割，会对剩下的小块进行计算
        for i in range(kh+skip_h):   # 垂直切割
            # 如果是最后是一个小块
            if i ==kh: #如果是最后一块,且最后一块是可分割的小块，隐含skip==1的条件
                    hi = lkh*32+260
                    flag_y_h=lkh # 最后一小块block中单个ROI的在h上的k值,既单个OPT的h尺寸-1
            else:
                hi = max_k*32+260
                flag_y_h=max_k
            flag_y_h= 2*(flag_y_h+1) # 该block对应的k值,既DPTS的h尺寸
            for j in range(kw+skip_w):
                if j ==kw :  #如果是最后的块,且最后一块是可分割的小块，隐含skip==1的条件
                    wi = lkw *32+260
                    flag_x_w=lkw  # 同上
                else:
                    wi = max_k*32+260
                    flag_x_w=max_k
                flag_x_w= 2*(flag_x_w+1) #同上
                print('%d row %d line, block size: (%d,%d),loc:(%d,%d)'%(j,i,wi,hi,w,h))                   
                block = slide.read_region((w,h),0,(wi,hi))

                    # 拼接pts到fpm中  
                print("fpm loc: (%d,%d,%d,%d)"%(flag_x,flag_x+flag_x_w,flag_y,flag_y+flag_y_h))
                dpt = self.get_dpt(block,wi,hi)

                final_probability_map[flag_x:flag_x+flag_x_w, flag_y:flag_y+flag_y_h] = dpt
                flag_x += flag_x_w
                w += wi
            flag_y+=flag_y_h
            flag_x=0
            w= 0 #回到从0开始  
            h += hi # h位移

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
save_npy='/root/workspace/renqian/0929/scannet/11_5'
post = PostScan(scannet=model,save=save_npy)
# 增加断点保存功能
saved=[]
for parent, dirnames, filenames in os.walk(save_npy):
    for filename in filenames:
        saved.append(filename.rstrip('_fpm.npy'))
print('saved:',saved)  
for slide_path in slide_list: 
    filename=os.path.basename(slide_path).rstrip('.tif')
#     print(filename in saved)
    if filename in saved:
        continue
    print(slide_path)
    sttime=time.time()
    final_probability_map=post.finalprobmap(slide_path,max_k=50)
    end=time.time()
    print('total time %f'%(end-sttime))