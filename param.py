#!/usr/bin/env python
"""
@Author: XSH
@Date: 2023-12-02 
@Description:  
"""
# /home/featurize/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData

# brats_path = r"../data/unprocessed/Brats"
brats_path = r"E:\deep learning\datasets\BrainTumour\brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

hgg_csv = "20hgg.csv"
lgg_csv = "20lgg.csv"

# /home/featurize/work/BrainMriSeg/BraTS2Dpreprocessing-master/train.csv
train_csv_path = 'train.csv'
# /home/featurize/work/BrainMriSeg/BraTS2Dpreprocessing-master/test.csv
test_csv_path = 'test.csv'

epochs = 100
test_size = 0.33

# /home/featurize/data/processed/2D
outputTrainImg_path = r"../data/processed/2D/trainImage"
outputTrainMask_path = r"../data/processed/2D/trainMask"

outputTestImg_path = r"../data/processed/2D/testImage"
outputTestMask_path = r"../data/processed/2D/testMask"

IMG_PATH = r'..\..\data\processed\2D\trainImage\*'
MASK_PATH = r'..\..\data\processed\2D\trainMask\*'

IMG_PATH_TEST = r"..\..\data\processed\2D\testImage\*"
MASK_PATH_TEST = r"..\..\data\processed\2D\testMask\*"


outputTrainImg_path3d = r"../data/processed/3D/trainImage"
outputTrainMask_path3d = r"../data/processed/3D/trainMask"

outputTestImg_path3d = r"../data/processed/3D/testImage"
outputTestMask_path3d = r"../data/processed/3D/testMask"

IMG_PATH3d = r'..\..\data\processed\3D\trainImage\*'
MASK_PATH3d = r'..\..\data\processed\3D\trainMask\*'

IMG_PATH_TEST3d = r"..\..\data\processed\3D\testImage\*"
MASK_PATH_TEST3d = r"..\..\data\processed\3D\testMask\*"

# 查看pkl文件信息
# import pickle
# with open(r'E:\deep learning\BrainMriSeg\model2D\UNet2D_BraTs-master\models\Brats_Unet_woDS_1\args.pkl', 'rb') as f:
#     data = pickle.load(f)
