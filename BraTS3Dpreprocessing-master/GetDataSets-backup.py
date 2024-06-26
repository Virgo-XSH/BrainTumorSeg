#!/usr/bin/env python
"""
@Author: XSH
@Date: 2024-03-30 
@Description:  
"""
# !/usr/bin/env python
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

BLOCKSIZE = (146, 192, 152)  # 每个分块的大小

def generate(outputImg_path, outputMask_path, pathhgg_list, flag='train'):
    if not os.path.exists(outputImg_path):
        makedir(outputImg_path)
    if not os.path.exists(outputMask_path):
        makedir(outputMask_path)

    for subsetindex in range(len(pathhgg_list)):

        print(pathhgg_list[subsetindex])
        # 1、读取数据
        brats_subset_path = param.brats_path + "/" + str(pathhgg_list[subsetindex]) + "/"
        flair_image = brats_subset_path + str(pathhgg_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t2_name
        mask_image = brats_subset_path + str(pathhgg_list[subsetindex]) + mask_name
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
        flair_array = sitk.GetArrayFromImage(flair_src)
        t1_array = sitk.GetArrayFromImage(t1_src)
        t1ce_array = sitk.GetArrayFromImage(t1ce_src)
        t2_array = sitk.GetArrayFromImage(t2_src)
        mask_array = sitk.GetArrayFromImage(mask)

        # 2、人工加入切片
        # myblackslice = np.zeros([240, 240])
        # flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        # flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        # flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        # flair_array = np.insert(flair_array, flair_array.shape[0], myblackslice, axis=0)
        # flair_array = np.insert(flair_array, flair_array.shape[0], myblackslice, axis=0)
        # t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        # t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        # t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        # t1_array = np.insert(t1_array, t1_array.shape[0], myblackslice, axis=0)
        # t1_array = np.insert(t1_array, t1_array.shape[0], myblackslice, axis=0)
        # t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        # t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        # t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        # t1ce_array = np.insert(t1ce_array, t1ce_array.shape[0], myblackslice, axis=0)
        # t1ce_array = np.insert(t1ce_array, t1ce_array.shape[0], myblackslice, axis=0)
        # t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        # t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        # t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        # t2_array = np.insert(t2_array, t2_array.shape[0], myblackslice, axis=0)
        # t2_array = np.insert(t2_array, t2_array.shape[0], myblackslice, axis=0)
        # mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        # mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        # mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        # mask_array = np.insert(mask_array, mask_array.shape[0], myblackslice, axis=0)
        # mask_array = np.insert(mask_array, mask_array.shape[0], myblackslice, axis=0)

        # 3、对四个模态分别进行标准化
        flair_array_nor = normalize(flair_array)
        t1_array_nor = normalize(t1_array)
        t1ce_array_nor = normalize(t1ce_array)
        t2_array_nor = normalize(t2_array)

        # 4、裁剪
        flair_crop = crop_ceter(flair_array_nor, 192, 152)
        t1_crop = crop_ceter(t1_array_nor, 192, 152)
        t1ce_crop = crop_ceter(t1ce_array_nor, 192, 152)
        t2_crop = crop_ceter(t2_array_nor, 192, 152)
        mask_crop = crop_ceter(mask_array, 192, 152)

        # 5、分块处理
        # patch_block_size = BLOCKSIZE
        # numberxy = patch_block_size[1] # 192
        # numberz = 8  # patch_block_size[0]
        # # width = np.shape(flair_crop)[1]
        # # height = np.shape(flair_crop)[2]
        # # imagez = np.shape(flair_crop)[0]
        # block_width = np.array(patch_block_size)[1]
        # block_height = np.array(patch_block_size)[2]
        # blockz = np.array(patch_block_size)[0]
        # stridewidth = (width - block_width) // numberxy
        # strideheight = (height - block_height) // numberxy
        # stridez = (imagez - blockz) // numberz
        # step_width = width - (stridewidth * numberxy + block_width)
        # step_width = step_width // 2
        # step_height = height - (strideheight * numberxy + block_height)
        # step_height = step_height // 2
        # step_z = imagez - (stridez * numberz + blockz)
        # step_z = step_z // 2

        # hr_samples_flair_list = []
        # hr_samples_t1_list = []
        # hr_samples_t1ce_list = []
        # hr_samples_t2_list = []
        # hr_mask_samples_list = []
        # patchnum = []
        # for z in range(step_z, numberz * (stridez + 1) + step_z, numberz):
        #     for x in range(step_width, numberxy * (stridewidth + 1) + step_width, numberxy):
        #         for y in range(step_height, numberxy * (strideheight + 1) + step_height, numberxy):
        #             if np.max(mask_crop[z:z + blockz, x:x + block_width, y:y + block_height]) != 0:
        #                 print("切%d" % z)
        #                 patchnum.append(z)
        #                 hr_samples_flair_list.append(flair_crop[z:z + blockz, x:x + block_width, y:y + block_height])
        #                 hr_samples_t1_list.append(t1_crop[z:z + blockz, x:x + block_width, y:y + block_height])
        #                 hr_samples_t1ce_list.append(t1ce_crop[z:z + blockz, x:x + block_width, y:y + block_height])
        #                 hr_samples_t2_list.append(t2_crop[z:z + blockz, x:x + block_width, y:y + block_height])
        #                 hr_mask_samples_list.append(mask_crop[z:z + blockz, x:x + block_width, y:y + block_height])
        # samples_flair = np.array(flair_crop).reshape(
        #     (len(flair_crop), 146, 192, 152))
        # samples_t1 = np.array(t1_crop).reshape((len(t1_crop), 146, 192, 152))
        # samples_t1ce = np.array(t1ce_crop).reshape(
        #     (len(t1ce_crop), 146, 192, 152))
        # samples_t2 = np.array(t2_crop).reshape((len(t2_crop), 146, 192, 152))

        # samples, imagez, height, width = np.shape(samples_flair)[0], np.shape(samples_flair)[1], \
        #     np.shape(samples_flair)[2], np.shape(samples_flair)[3]
        # 5、合并和保存
        # samples = np.zeros((4, 192, 192, 4), np.float64)
        # for j in range(len(pathhgg_list)):
        """
        merage 4 model image into 4 channel (imagez,width,height,channel)
        """
        fourmodelimagearray = np.zeros((146, 192, 152, 4), np.float64)

        tempName = str(pathhgg_list[subsetindex]).replace("Training", flag)

        filepath1 = outputImg_path + "/" + tempName + ".npy"
        filepath = outputMask_path + "/" + tempName + ".npy"

        flairimage = flair_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 0] = flairimage
        t1image = t1_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 1] = t1image
        t1ceimage = t1ce_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 2] = t1ceimage
        t2image = t2_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 3] = t2image
        np.save(filepath1, fourmodelimagearray)


        wt_tc_etMaskArray = np.zeros((146, 192, 152, 3), np.uint8)
        mask_one_sample = mask_crop
        WT_Label = mask_one_sample.copy()
        WT_Label[mask_one_sample == 1] = 1.
        WT_Label[mask_one_sample == 2] = 1.
        WT_Label[mask_one_sample == 4] = 1.
        TC_Label = mask_one_sample.copy()
        TC_Label[mask_one_sample == 1] = 1.
        TC_Label[mask_one_sample == 2] = 0.
        TC_Label[mask_one_sample == 4] = 1.
        ET_Label = mask_one_sample.copy()
        ET_Label[mask_one_sample == 1] = 0.
        ET_Label[mask_one_sample == 2] = 0.
        ET_Label[mask_one_sample == 4] = 1.
        wt_tc_etMaskArray[:, :, :, 0] = WT_Label
        wt_tc_etMaskArray[:, :, :, 1] = TC_Label
        wt_tc_etMaskArray[:, :, :, 2] = ET_Label
        np.save(filepath, wt_tc_etMaskArray)
    print("Done!")

if __name__ == '__main__':
    trainList1, testList1 = split_test(param.hgg_csv, flag="hgg")
    trainList2, testList2 = split_test(param.lgg_csv, flag="lgg")

    trainList = trainList1 + trainList2
    testList = testList1 + testList2
    trainList.sort()
    testList.sort()
    generate(param.outputTrainImg_path3d, param.outputTrainMask_path3d, trainList, "train")
    generate(param.outputTestImg_path3d, param.outputTestMask_path3d, testList, "test")
