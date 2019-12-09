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
import pdb
class PostScan():
    def __init__(self, scannet,transform,save_path=None, dense_coefficient=2, maxpools=5, stride=2):
        '''

        :param scannet: scannet 模型
        :param dpt_size: 输出dpt的大小
        :param dense_coefficient: DPTs/OPTs的尺寸比例
        :param maxpools: 最大池化层数
        :param stride: 模型步长
        '''
        self.model = scannet
        self.alpha = int(dense_coefficient) # ratio between the size of DPTs and the size of OPT
        self.sf = int(math.pow(stride, maxpools))  # 求出的Sf是FCN的滑动步长 inner stride of scannet
        self.sd = int(self.sf / self.alpha)  # 偏移量Sd
        self.lf = 244  # 输入Scannet的概率图
        self.transform = transform
        self.path = save_path


    def inner_scan(self,roi,h,w):
        '''(测试通过)
        设定的roi是PIL.Image类
        Lr = Lf + (Lp -1) * Sf; Sr = Sf *Lp
        假设Lr = 2868，Sf=32，Lf=244，则Lp=83(吻合，此处ok),此时opt大小为LpXLpX2,经过softmax转换成LpXLpX1的p值
        :param roi: 单个ROI区域
        :param h,w:输入ROI区域的尺寸
        :return: opt矩阵
        '''
        hp, wp = int((h-self.lf)/self.sf+1), int((w-self.lf)/self.sf+1)
#         print('hp:%d,wp:%d'%(hp,wp))
        opt = torch.zeros((hp,wp)) #初始化opt
        x, y=0, 0 #初始化坐标
        for i in range(hp):
            for j in range(wp):
                x += i*32
                y += j*32
                sroi=roi.crop((x, y, x+244, y+244)).convert('RGB')
                sroi=transforms.ToTensor()(sroi)
                sroi=Variable(sroi.type(torch.cuda.FloatTensor)).unsqueeze(dim=0)
#                 print(sroi.shape)
                temp = self.model(sroi)
                opt[i, j] = F.softmax(temp)[:,1].cpu().detach() #切割244X244大小的图像作为输入到model并计算
        return opt
    
    def test_inner_scan(self,h,w):
        hp, wp = int((h-self.lf)/self.sf+1), int((w-self.lf)/self.sf+1)
#         print('hp:%d,wp:%d'%(hp,wp))
        opt = torch.zeros((hp,wp)) #初始化opt
        x, y=0, 0 #初始化坐标
        for i in range(hp):
            for j in range(wp):
                x += i*32
                y += j*32
                opt[i,j]= torch.zeros((1))
        return opt


    def get_dpt(self, block,hi,wi):
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
        def reconstruction(dpt,opts):
            '''
            After scan image, we can reconstuct DPT.
            :return:
            '''
            H, W = dpt.shape
            for h_ in range(H):
                for w_ in range(W):
                    i = h_ % self.alpha
                    j = w_ % self.alpha
                    h = int(h_ / self.alpha)
                    w = int(w_ / self.alpha)
                    dpt[h_, w_] = opts[i, j][h, w]
            return dpt
        x, y = 0, 0
        hp, wp = int(hi - self.sd *(self.alpha - 1)), int(wi - self.sd * (self.alpha - 1))  #计算ROI区域大小
        ho, lo = int((hp - self.lf) / self.sf) + 1, int((wp - self.lf) / self.sf) + 1 #计算ROI区域的Lp值
        opts = torch.zeros((self.sd,self.sd,ho,lo)) # 初始化opts矩阵
        print('opts.shape')
        print(opts.shape)
        dpt = torch.zeros((self.alpha*ho,self.alpha*lo))  # 初始化dpts矩阵
        print('dpt.shape')
        print(dpt.shape)
        for i in range(self.sd):
            for j in range(self.sd):
                x += i * self.sd
                y += j * self.sd
                roi = block.crop((x, y, x+hp, y+wp))  # Image.crop(x,y,hp,wp)
                opts[i, j] = self.inner_scan(roi,hp,wp)  # 计算ROI区域的P[i,j]
        dpt = reconstruction(dpt,opts)
        return dpt,dpt.shape


    
    def test_finalprobmap(self,slide_path, max_k=82,save=None,num_worker=None):
        '''
        max_k = 82按照paper给的结果换算而成，Lp=83，n=83-1=82,用于调控block的大小
        将wsi slide分成多块，然后分开对每个小块求DPT，这里需要提供一个方法解决分块问题，将求得的DPTs缝合起来就得到这张图的最后概率密度图
        Hp代表的是stitched probability map， 或许等于block大小，这里需要测试一下
        设ROI的size为 Lr = 244 + 32k,则 Block的size= ROI+16=260+32k
        如何处理小块？想到的解决方案有：方法1：需要填充，方法2：小块单独计算，此处选择小块单独计算。
        如何提高计算速度：计算量较大，建议使用多线程(后面将补充该功能)
        :param wsi: 输入整张wsi
        :return:fpm 返回最后的概率值
        '''
        slide = openslide.open_slide(slide_path)
        h, w = slide.dimensions
        print('slide size: (%d,%d)'%(h,w))
        kh, kw = int(h/(260+max_k*32)),int(w/(260+max_k*32))  #求w,h对应的k值，既能够划分为max_k的block的块数，会有剩余的小块
        lh, lw = h%(260+max_k*32), w%(260+max_k*32) #求小块的大小
        lkh,lkw=int((lh-260)/32), int((lw-260)/32) # 求小块的k值
        x,y=0,0 # 初始化最后概率图的的大小
        h= 0
        dpts={} #将所有的dpt用dpts保存
        # 将图像按照max_k的值进行切割，可能会对剩下的小块进行计算
        for i in range(kh+1):
            h += i * (max_k*32+260)
            if i ==kh:  # 如果是最后是一个小块
                hi = lh*32+260
            else:
                hi = max_k*32+260
            w= 0
            x += h
            for j in range(kw+1):
                w += j *(max_k*32+ 260)
                if j ==kw:  
                    wi = lw *32+260
                else:
                    wi = max_k*32+260
                if lh == 0 and lw ==0:  #如果顺利完成切割则不用计算小块
                    i -=1
                    j -=1
                    break
                if num_worker ==None: # 将添加多线程
#                     block = slide.read_region((h,w),0,(hi,wi))
                    dpt,(h,w)=self.test_get_dpt(hi,wi)
                    dpts[(i,j)]=(dpt,h,w)
                    y += w
        fpm = torch.zeros((x,y))  # 初始化fpm
        print(fpm.shape)
        # 拼接pts到fpm中b  
        for n in range(i+1):
            for m in range(j+1):
                dpt,h,w = dpts[(n,m)]
                fpm[x:x+h,y:y+w]=dpt
            x+=h
            y=0   
        return fpm
        
    def finalprobmap(self,slide_path, max_k=82,save=None,num_worker=None):
        '''
        max_k = 82按照paper给的结果换算而成，Lp=83，n=83-1=82,用于调控block的大小
        将wsi slide分成多块，然后分开对每个小块求DPT，这里需要提供一个方法解决分块问题，将求得的DPTs缝合起来就得到这张图的最后概率密度图
        Hp代表的是stitched probability map， 或许等于block大小，这里需要测试一下
        设ROI的size为 Lr = 244 + 32k,则 Block的size= ROI+16=260+32k
        如何处理小块？想到的解决方案有：方法1：需要填充，方法2：小块单独计算，此处选择小块单独计算。
        如何提高计算速度：计算量较大，建议使用多线程(后面将补充该功能)
        :param wsi: 输入整张wsi
        :return:fpm 返回最后的概率值
        '''
        slide = openslide.open_slide(slide_path)
        h, w = slide.dimensions
        print('slide size: (%d,%d)'%(h,w))
        kh, kw = int(h/(260+max_k*32)),int(w/(260+max_k*32))  #求w,h对应的k值，既能够划分为max_k的block的块数，会有剩余的小块
        lh, lw = h%(260+max_k*32), w%(260+max_k*32) #求小块的大小
        lkh,lkw=int((lh-260)/32), int((lw-260)/32) # 求小块的k值
        x,y=0,0 # 初始化最后概率图的的大小
        h= 0
        dpts={} #将所有的dpt用dpts保存
        # 将图像按照max_k的值进行切割，可能会对剩下的小块进行计算
        for i in range(kh+1):
            h += i * (max_k*32+260)
            if i ==kh:  # 如果是最后是一个小块
                hi = lh*32+260
            else:
                hi = max_k*32+260
            w= 0
            x += h
            for j in range(kw+1):
                w += j *(max_k*32+ 260)
                if j ==kw:  
                    wi = lw *32+260
                else:
                    wi = max_k*32+260
                if lh == 0 and lw ==0:  #如果顺利完成切割则不用计算小块
                    i -=1
                    j -=1
                    break
                if num_worker ==None: # 将添加多线程
                    block = slide.read_region((h,w),0,(hi,wi))
                    dpt,h,w=self.test_get_dpt(block,hi,wi)
                    dpts[(i,j)]=(dpt,h,w)
                    y += w
        fpm = torch.zeros((x,y))  # 初始化fpm
        print(fpm.shape)
        # 拼接pts到fpm中b  
        for n in range(i+1):
            for m in range(j+1):
                dpt,h,w = dpts[(n,m)]
                fpm[x:x+h,y:y+w]=dpt
            x+=h
            y=0   
        return fpm

    def test(self,slide_path, max_k=82):
        slide = openslide.open_slide(slide_path)
        h, w = slide.dimensions
        kh, kw = (h-260)/self.sf,(w-260)/self.sf  #求w,h对应的k的值
        nh, nw = int(kh/max_k), int(kw/max_k)  # 分别求w,h上分割的个数
        h, w= 0, 0
        dpts={}
        hi = max_k*32+260
        wi = max_k*32+260
        block = slide.read_region((h,w),0,(max_k*32+260,max_k*32+260))
        print((h,w),0,(max_k*32+260,max_k*32+260))
        dpts['%d_%d'%(0,0)]=self.get_dpt(block,hi,wi)
        return dpts
                
        
import glob,os
test_slide_folder = '/root/workspace/dataset/CAMELYON16/testing/images/'
test_slide_annotation_folder = '/root/workspace/dataset/CAMELYON16/testing/lesion_annotations/'

train_slide_folder = '/root/workspace/dataset/CAMELYON16/training/*'
train_slide_annotation_folder = '/root/workspace/dataset/CAMELYON16/training/lesion_annotations/'
slide_list = glob.glob(os.path.join(test_slide_folder, '*.tif'))
slide_list.sort()
print('total slide : %d' % len(slide_list))
test_slide=slide_list[0]
pth = '/root/workspace/renqian/0929/save/camelyon16/scannet_train_MSE_NCRF_40w_patch_256/2019-10-21_08-33-34/hardmine_0_epoch_9_type_train_model.pth'
model = Scannet().cuda()
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(pth)['model_state'])
transform1=transforms.Compose([transforms.ToTensor()])
post = PostScan(scannet=model,transform=transform1)
fpms=post.test(test_slide)

print('fpms.shape:',fpms.shape)
