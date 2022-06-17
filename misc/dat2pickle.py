# ディレクトリ内の全てのdatファイルを２次元のnumpy行列としてpickleに保存
# datファイルのフォーマットは、「１列目が経度、２列目が緯度、３列目が観測値」
# また、CSVのようにカンマ区切りであるとする

import numpy as np
import os
import glob
import pdb
import pickle
import pandas as pd

# 点のオブジェクト
class point(object) :
    def __init__(self, lon, lat, value):
        self.lon = lon
        self.lat = lat
        self.value = value
    
    def isAt(self, lon, lat):
        if (self.lon == lon) and (self.lat == lat) :
            return True
        else:
            return False


# 点群のオブジェクト
class pointsObject(object) :
    def __init__ (self, lons, lats, values) :
        # 点群データ
        self.points = [point(lon, lat, val) for lon,lat,val in zip(lons, lats, values)]

        # 重複を削除
        self.uniqLons = np.sort(np.unique(lons))
        lonNum = self.uniqLons.shape[0]
        self.uniqLats = np.sort(np.unique(lats))
        latNum = self.uniqLats.shape[0]

        self.lons = np.tile(self.uniqLons[:,np.newaxis],[latNum,1]).flatten()
        self.lats = np.tile(self.uniqLats[:,np.newaxis],[1,lonNum]).flatten()

        # 行列のサイズ
        self.defSize= [latNum, lonNum]
    
    # 行列に変換して出力
    # 海洋部のマスクもほしい場合は getMask=True
    def toMatrix(self, getMask=False) :
        # 空の行列を作って, size of matrix is set so as to avoid duplicate same pixels
        matrix = np.zeros(self.defSize)

        # インデックスのグリッド
        grid = np.meshgrid(range(self.defSize[1]),range(self.defSize[0]))
        gridX = grid[0].flatten()
        gridY = grid[1].flatten()[::-1]

        if getMask:
            mask = np.zeros(self.defSize)
        
        # 緯度経度が一致する点の値を行列に代入
        for p in self.points:
            x = gridX[(p.lon == self.lons) & (p.lat == self.lats)]
            y = gridY[(p.lon == self.lons) & (p.lat == self.lats)]
            matrix[y,x] = p.value
            if getMask:
                mask[y,x] = 1 # マスクを更新（値のある点が１）
    
        # for x, y, lon, lat in zip(gridX, gridY, self.lons, self.lats):
        #     for p in self.points:
        #         if p.isAt(lon,lat): # 緯度経度が一致したら
        #             matrix[y,x] = p.value # 値を代入
        #             print(f"\r making matrix : point({x},{y}) is done", end="")
        #             if getMask:
        #                 mask[y,x] = 1 # マスクを更新（値のある点が１）
        #             break
        

        if getMask:
            return matrix, mask

        return matrix



if __name__ == '__main__':
    path = "../data/eq_data/"
    
    # datファイルのあるディレクトリ
    directoryName = f'{path}{os.sep}2013_Sv05s_LL{os.sep}'

    # 観測点の情報が入ったxlsxファイル(観測点のマスクを作るのに使用)
    excelPath = f"{path}{os.sep}site_schema_20200322A.xlsx"

    # 全てのdatファイルのパスを取得
    datFilePaths = glob.glob(directoryName + '*.dat')
    fileNum = len(datFilePaths)
    
    #==================================
    # datファイルから点群のリストを取得
    #==================================
    pointsList = []
    # ファイル名のリスト
    fileNames = []

    for ind,path in enumerate(datFilePaths):
        print(f"\r load datFile {ind+1}/{fileNum}",end="")
        lons = np.loadtxt(path, np.float32, delimiter=',', usecols=0) # 経度
        lats = np.loadtxt(path, np.float32, delimiter=',', usecols=1) # 緯度
        spectrums = np.loadtxt(path, np.float32, delimiter=',', usecols=2) # 応答スペクトル
        # 点群のオブジェクトとして格納
        pointsList.append(pointsObject(lons, lats, spectrums))
        # ファイル名を格納
        fileNames.append(path.split(os.sep)[-1])
    print("")

    #==================================
    # 観測点のマスク作成
    #==================================
    print("\r make observe-mask...", end="")
    excel = pd.read_excel(excelPath)
    excelLonLat = np.array([excel['lon'].values, excel['lat'].values])
    # 緯度・経度の範囲を取得
    sampleP = pointsList[0]
    minLon = np.min(sampleP.lons)
    minLat = np.min(sampleP.lats)
    maxLon = np.max(sampleP.lons)
    maxLat = np.max(sampleP.lats)
    # 緯度・経度の区切り幅を取得
    disLon = sampleP.uniqLons[1] - sampleP.uniqLons[0]
    disLat = sampleP.uniqLats[1] - sampleP.uniqLats[0]

    # 使用する観測点をnp.whereで選択
    keepInds = [] 

    keepLons = np.concatenate([
        np.where(excelLonLat[0]>=minLon-disLon/2)[0],
        np.where(excelLonLat[0]<=maxLon+disLon/2)[0]
    ])
    uni,counts_lon = np.unique(keepLons,return_counts=True)
    keepInds.append(uni[counts_lon>=2])

    keepLats = np.concatenate([
        np.where(excelLonLat[1]>=minLat-disLat/2)[0],
        np.where(excelLonLat[1]<=maxLat+disLat/2)[0]
    ])
    uni,counts_lat = np.unique(keepLats,return_counts=True)
    keepInds.append(uni[counts_lat>=2])

    uni,counts = np.unique(np.concatenate(keepInds),return_counts=True)
    keepInds = uni[counts>=2]

    excelLonLat = excelLonLat[:, keepInds].transpose() # [N,2] 観測点
    
    # 経度、緯度のリストからのインデックスのリストに
    # lon,lat ->  indexs x,y
    # ※完全一致する緯度経度はないので、距離の近いセルを観測点とする
    obsXInds = []
    obsYInds = []
    for p in excelLonLat:
        x = np.abs(sampleP.uniqLons-p[0]).argmin()
        y = np.abs(sampleP.uniqLats-p[1]).argmin()
        obsXInds.append(x)
        obsYInds.append(y)
    # 縦方向（緯度）は降順に並ぶので大きさ反転
    obsYInds = np.abs(np.array(obsYInds) - sampleP.defSize[1])

    # 観測点のマスク
    obsMask = np.zeros(sampleP.defSize)
    for x, y in zip(obsXInds,obsYInds):
        obsMask[y, x] = 1
    
    # pickleに保存
    pickle.dump(obsMask, open("observeMask.pickle","wb"))

    # 画像にも保存
    # import cv2
    # cv2.imwrite("obsMask.png",obsMask*255)
    
    print("\r make observe-mask : Done")

    #==================================
    # 各点群を行列（画像）に変換
    #==================================
    matrixes = []
    existMasks = []
    for i,ps in enumerate(pointsList):
        print(f"make matrix {i+1}/{len(pointsList)}")
        mat, mask = ps.toMatrix(True)
        matrixes.append(mat)
        existMasks.append(mask)

    #==================================
    # pickleに保存
    #==================================
    datas = {
        "matrixes": matrixes,
        "fileName": fileNames
    }

    pickle.dump(datas,open("datas.pickle", "wb"))
    pickle.dump(existMasks,open("existMask.pickle","wb"))

