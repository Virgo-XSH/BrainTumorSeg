#!/usr/bin/env python
"""
@Author: XSH
@Date: 2024-03-25 
@Description:  
"""
import SimpleITK as sitk
import skimage.io as io
import numpy as np

def crop_ceter(img,croph,cropw):
    #for n_slice in range(img.shape[0]):
    height,width = img[0].shape
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)
    return img[:,starth:starth+croph,startw:startw+cropw]

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

# 显示一个系列图
def show_img1(data):
    for i in range(data.shape[0]):
        io.imshow(data[i, :, :], cmap='gray')
        print(i)
        io.show()

# 单张显示
def show_img2(ori_img):
    io.imshow(ori_img[74], cmap='gray')
    io.show()

path1=r"D:\project\BrainMriSeg\data\processed\BraTS20_train_001_77.npy"
data1=np.load(path1)
data1=data1.transpose((2,0,1))
show_img1(data1)
path2=r"D:\project\BrainMriSeg\data\processed\BraTS20_trainMask_001_77.npy"
data2=np.load(path2)
print(data2.shape)
io.imshow(data2, cmap='gray')
io.show()

# data = read_img(data_path)
# show_img2(data)



# data_path = r'E:\deep learning\datasets\BrainTumour\brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1.nii' #文件路径
# t1_src = sitk.ReadImage(data_path, sitk.sitkInt16)
# t1_array = sitk.GetArrayFromImage(t1_src)
# t1_crop = crop_ceter(t1_array, 160, 160)
# show_img2(t1_crop)


