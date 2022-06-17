import pickle
import csv
import sys
import os
import numpy as np
import glob
import pdb

from PIL import Image
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument('dirPath',type=str,help='path of Directory')
    # parser.add_argument('pickleName',type=str,help='name of output pickle file')
    parser.add_argument('-type','--type',type=str,default='csv', help='Data type, e.g. \'csv\',\'png\' ', choices=['csv','png','jpg'])
    parser.add_argument('-maxI','--maxI',default="", help='pickle化する画像の最大輝度値 (default=最大値)')
    parser.add_argument('-mask','--mask',action='store_true',help="Flag for mask data")

    return  parser.parse_args()

def datasetUpdate(dirPath,thre=0.2):
    # データセットのディレクトリ上で
    # train.pickle, valid.pickle, test.pickle にXYの平均・分散・共分散（5次元）のベクトルを追加する
    data = ["train.pickle", "valid.pickle", "test.pickle"]

    for d in data:
        # pdb.set_trace()

        path = os.path.join(dirPath,d)
        pickleData = pickle.load(open(path,"rb"))
        imgs = pickleData["images"]
        shape = imgs.shape
        X = np.array([[i for i in range(shape[2])] for _ in range(shape[1])])
        Y = np.array([[shape[1]-i-1 for _ in range(shape[2])] for i in range(shape[1])])
        X = X[:,:,np.newaxis]
        Y = Y[:,:,np.newaxis]
        vec = []

        for img in imgs:
            # pdb.set_trace()
            threX = X[img>thre]
            threY = Y[img>thre]
            
            meanX = np.mean(threX)
            meanY = np.mean(threY)
            covXY = np.cov(threX,threY) # 分散共分散行列
            
            vec.append([meanX,meanY,covXY[0,0],covXY[1,1],covXY[1,0]])

        newData = {
            "images":imgs,
            "labels":pickleData["labels"],
            "vector":np.array(vec),
            "thre":thre
            }
        
        pickle.dump(newData,open(path,"wb"))


if __name__=="__main__":
    # Parse command-line arguments
    args = parse_args()
    # pdb.set_trace()
    datasetUpdate(args.dirPath,thre=0.4)
    sys.exit()

    isMask = args.mask
    extension = "*" if args.type=="" else "*." + args.type
    dsPath = os.path.abspath(args.dirPath)
    dirList = ["train","train_mask","valid","valid_mask","test","test_mask"]

    for folder in dirList:
        folder = os.path.join(dsPath,folder)
        fileList = [os.path.abspath(p) for p in glob.glob(os.path.join(folder,extension))]
        images = []
        labels = []
        for i,p in enumerate(fileList):
            print("\r processing:{}/{}".format(i+1,len(fileList)),end="")

            # ファイル名をラベルとして保存
            lab = p.split(os.sep)[-1].split(".")[-2]
            labels.append(lab)

            if args.type=='csv':
                with open(p) as f:
                    reader = csv.reader(f)
                    img = [[float(col) for col in row] for row in reader]
            elif args.type=='png' or args.type=='jpg':
                img = np.array(Image.open(p).convert('L'))

            images.append(np.array(img))

        # 正規化
        images = np.array(images)
        if args.maxI == "": # 最小値を引いて最大値で割る正規化（0~1の範囲に）
            images = images - np.min(images)
            images = images / np.max(images)
        else:
            images = images/float(args.maxI)

        print("")
        images = images[:,:,:,np.newaxis]
        
        # dump data
        fileName = folder.split(os.sep)[-1] # フォルダ名をファイル名に
        fileName = fileName.split(".")[0] + ".pickle" # 拡張子をpickleに

        if "mask" in fileName:
            data = images
        else:
            data = {
                "images":images,
                "labels":labels
            }
        pickle.dump(data,open(os.path.join(dsPath,fileName),"wb"))

