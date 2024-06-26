#!/usr/bin/env python
"""
@Author: XSH
@Date:
@Description:  
"""
import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
import random
import param

flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"

# 划分数据集
def split_test(file, rate=0.2, flag='hgg'):
    """
    df_flow = pd.read_csv(file, header=None)

    # 6:2:2
    train_len = int(len(df_flow) * 0.6)
    test_len = int(len(df_flow) * 0.2)

    # split the dataframe
    idx = list(df_flow.index)
    random.seed(17)
    random.shuffle(idx)  # 将index列表打乱
    df_train = df_flow.loc[idx[:train_len]]
    df_test = df_flow.loc[idx[train_len:train_len + test_len]]
    df_valid = df_flow.loc[idx[train_len + test_len:]]  # 剩下的就是valid

    # output
    df_train.to_csv(input_path + 'train.txt', header=False, index=False, sep='\t')
    df_test.to_csv(input_path + 'test.txt', header=False, index=False, sep='\t')
    df_valid.to_csv(input_path + 'valid.txt', header=False, index=False, sep='\t')
    """
    df_flow = pd.read_csv(file, header=None)
    test_len = int(round((len(df_flow) * rate)))
    train_len = len(df_flow) - test_len
    idx = list(df_flow.index)
    random.seed(25)
    random.shuffle(idx)  # 将index列表打乱
    df_train = df_flow.loc[idx[:train_len]]
    df_test = df_flow.loc[idx[train_len:(train_len + test_len)]]

    df_train_list = df_train.to_dict(orient='list')[0]
    df_test_list = df_test.to_dict(orient='list')[0]

    if (len(df_train_list) + len(df_test_list) != len(df_flow)):
        raise Exception('数据集划分错误')

    if (len(set(df_train_list)) < len(df_train_list)):
        raise Exception('训练集含重复项')

    if (len(set(df_test_list)) < len(df_test_list)):
        raise Exception('测试集含重复项')

    inter = set(df_train_list).intersection(df_test_list)
    if (len(inter) > 0):
        raise Exception('训练集与测试集有交集')

    tra = pd.DataFrame(df_train).sort_index(axis=0)
    tes = pd.DataFrame(df_test).sort_index(axis=0)
    tra.to_csv(flag + "-" + param.train_csv_path, header=False, index=False, sep='\t')
    tes.to_csv(flag + "-" + param.test_csv_path, header=False, index=False, sep='\t')
    return df_train_list, df_test_list

def makedir(createdDir):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
    abs_path = os.path.join(script_dir, createdDir)  # 将相对路径转换为绝对路径
    os.makedirs(abs_path, exist_ok=True)  # 创建文件夹

def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    # 有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)  # 限定范围numpy.clip(a, a_min, a_max, out=None)

    # 除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9  # 黑色背景区域
        return tmp

def crop_ceter(img, croph, cropw):
    # for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[:, starth:starth + croph, startw:startw + cropw]

def generate(outputImg_path, outputMask_path, path_list, flag='train'):
    if not os.path.exists(outputImg_path):
        makedir(outputImg_path)
    if not os.path.exists(outputMask_path):
        makedir(outputMask_path)

    for subsetindex in range(len(path_list)):
        brats_subset_path = param.brats_path + "/" + (path_list[subsetindex]) + "/"
        # 获取每个病例的四个模态及Mask的路径
        flair_image = brats_subset_path + (path_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + (path_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + (path_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + (path_list[subsetindex]) + t2_name
        mask_image = brats_subset_path + (path_list[subsetindex]) + mask_name
        # 获取每个病例的四个模态及Mask数据
        # SimpleITK图像顺序是x，y，z三个方向的大小
        """
        Width: 宽度，X轴，矢状面（Sagittal）
        Height: 高度，Y轴，冠状面（Coronal）
        Depth: 深度， Z轴，横断面（Axial）
        """
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
        # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
        # GetArrayFromImage得到的numpy维度是(z,y,x)
        # (155,240,240)
        flair_array = sitk.GetArrayFromImage(flair_src)
        t1_array = sitk.GetArrayFromImage(t1_src)
        t1ce_array = sitk.GetArrayFromImage(t1ce_src)
        t2_array = sitk.GetArrayFromImage(t2_src)
        mask_array = sitk.GetArrayFromImage(mask)
        # 对四个模态分别进行标准化,由于它们对比度不同
        flair_array_nor = normalize(flair_array)
        t1_array_nor = normalize(t1_array)
        t1ce_array_nor = normalize(t1ce_array)
        t2_array_nor = normalize(t2_array)
        # 裁剪(偶数才行)
        flair_crop = crop_ceter(flair_array_nor, 192, 160)
        t1_crop = crop_ceter(t1_array_nor, 192, 160)
        t1ce_crop = crop_ceter(t1ce_array_nor, 192, 160)
        t2_crop = crop_ceter(t2_array_nor, 192, 160)
        mask_crop = crop_ceter(mask_array, 192, 160)
        print((path_list[subsetindex]))
        # 切片处理,并去掉没有病灶的切片
        for n_slice in range(flair_crop.shape[0]):
            if np.max(mask_crop[n_slice, :, :]) != 0:
                maskImg = mask_crop[n_slice, :, :]
                #FourModelImageArray会在迭代时进行转置
                FourModelImageArray = np.zeros((flair_crop.shape[1], flair_crop.shape[2], 4), np.float64)
                flairImg = flair_crop[n_slice, :, :]
                flairImg = flairImg.astype(np.float64)
                FourModelImageArray[:, :, 0] = flairImg
                t1Img = t1_crop[n_slice, :, :]
                t1Img = t1Img.astype(np.float64)
                FourModelImageArray[:, :, 1] = t1Img
                t1ceImg = t1ce_crop[n_slice, :, :]
                t1ceImg = t1ceImg.astype(np.float64)
                FourModelImageArray[:, :, 2] = t1ceImg
                t2Img = t2_crop[n_slice, :, :]
                t2Img = t2Img.astype(np.float64)
                FourModelImageArray[:, :, 3] = t2Img

                tempName = str(path_list[subsetindex]).replace("Training", flag)

                imagepath = outputImg_path + "/" + tempName + "_" + str(n_slice) + ".npy"
                maskpath = outputMask_path + "/" + tempName + "_" + str(n_slice) + ".npy"
                np.save(imagepath, FourModelImageArray)  # (192,192,4) np.float64 dtype('float64')
                np.save(maskpath, maskImg)  # (192, 192) dtype('uint8') 值为0 1 2 4
    print("Done！")

if __name__ == '__main__':
    trainList1, testList1 = split_test(param.hgg_csv, flag="hgg")
    trainList2, testList2 = split_test(param.lgg_csv, flag="lgg")
    trainList = trainList1 + trainList2
    testList = testList1 + testList2
    trainList.sort()
    testList.sort()
    generate(param.outputTrainImg_path, param.outputTrainMask_path, trainList, "train")
    generate(param.outputTestImg_path, param.outputTestMask_path, testList, "test")

    # pathhgg_list = file_name_path(param.brats_path)
    # generate(param.outputTrainImg_path, param.outputTrainMask_path, pathhgg_list, "train")
    # generate(param.outputTestImg_path, param.outputTestMask_path, pathhgg_list, "test")
