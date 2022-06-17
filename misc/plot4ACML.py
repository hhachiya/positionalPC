import os
from PIL import Image
import numpy as np
import pdb
import pickle
import matplotlib.pylab as plt
import cv2
import pandas as pd
import glob
from libs.util import cmap
import copy

# setting path
lat = [30.220833, 38.8375]
lon = [128.93125, 141.60625]
test_data_path = "../data/quakeData-h04h05h09h10/"
sea_path = '../data/sea.png'
out_path = 'images/'
data_path = '../data/eq_data/'
scale_max = 8.0

#-------------------
# plot observation sites

# 各pickleデータのパス    
TEST_PICKLE = test_data_path+"test.pickle"
TEST_MASK_PICKLE = test_data_path+"test_mask.pickle"

# テストデータの読み込み
testImg = pickle.load(open(TEST_PICKLE,"rb"))["images"]
testLabel = pickle.load(open(TEST_PICKLE,"rb"))["labels"]
testMask = pickle.load(open(TEST_MASK_PICKLE,"rb"))
testMasked = testImg*testMask

mask = testMask[0,:,:,0]
sea = np.array(Image.open(sea_path))/255

# rad = 0.0
obs_site_img = 0.5 - 0.5*sea + mask
fig = plt.imshow(obs_site_img, cmap="gray", interpolation="None", extent=[lon[0], lon[1], lat[0], lat[1]])
#fig.axes.get_xaxis().set_visible(False)
#fig.axes.get_yaxis().set_visible(False)
plt.savefig(f"{out_path}/obs_points_r0_512x512.pdf",bbox_inches='tight')
#-------------------

#-------------------
# plot simulation data
files = sorted(glob.glob(data_path+"2013_Sv05s_LL/*.dat"))

# datファイルのループ
if 0:
    for ite,file in zip(range(len(files)),files):
        print(" \r ite:{}".format(ite),end="")

        # datファイルの読み込み
        df = pd.read_csv(file,names=["lon", "lat", "amp", "sid"])

        # 経度と緯度の値の格子
        lons = np.unique(df['lon'].values)
        lats = np.unique(df['lat'].values)[::-1]

        # マップ画像の初期化
        map = np.zeros([len(lats),len(lons)])
        sea = np.zeros([len(lats),len(lons)])

        for ind in range(len(df)):
            lonInd = int(np.where(df['lon'][ind]==lons)[0])
            latInd = int(np.where(df['lat'][ind]==lats)[0])

            map[latInd,lonInd] = df['amp'][ind]
            sea[latInd,lonInd] = 1

        # img
        if ite==0:
            fig = plt.imshow(sea, cmap="gray", interpolation="None", vmin=0, vmax=1)
            plt.savefig(f"{out_path}/sea.pdf",bbox_inches='tight')
            plt.clf()

        # img
        map_img = copy.deepcopy(map)
        map_img[sea == 0] = -1
        map_img[map_img<=0] -1
        cmap = plt.get_cmap("Reds")
        cmap.set_under('white')
        fig = plt.imshow(map_img, cmap=cmap, interpolation="None", vmin=0, vmax=scale_max)
        plt.savefig(f"{out_path}/{os.path.splitext(os.path.basename(file))[0]}_img.pdf",bbox_inches='tight')
        plt.clf()

        # map     
        map[sea == 0] = -1   
        cmap = plt.get_cmap("Reds")
        cmap.set_under('grey')
        fig = plt.imshow(map, cmap=cmap, interpolation="None", vmin=0, vmax=scale_max, extent=[lon[0], lon[1], lat[0], lat[1]])
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.colorbar(shrink=0.5)
        plt.savefig(f"{out_path}/{os.path.splitext(os.path.basename(file))[0]}_map.pdf",bbox_inches='tight')
        plt.clf()
#-------------------

#-------------------
# observation point
df = pd.read_csv(files[0],names=["lon", "lat", "amp", "sid"])
lons_df = np.unique(df['lon'].values)
lats_df = np.unique(df['lat'].values)[::-1]
p_df = np.array([df['lon'].values,df['lat'].values]).transpose() # [M,2] 全データ

obs_site_df = pd.read_excel(data_path+"site_schema_20200322A.xlsx")
print(f"number of site:{len(np.unique(obs_site_df.site_code))}")

# exist points
p_ex = np.array([obs_site_df['lon'].values, obs_site_df['lat'].values]) 

# interval of simulated points
dis_lon = lons_df[1]-lons_df[0]
dis_lat = lats_df[0]-lats_df[1]

# 観測値のxy軸の端
# start and end points of simulation
st = np.array([np.min(lons_df),np.min(lats_df)])
en = [np.max(lons_df),np.max(lats_df)]

# cut points over simulation range
keep_ind = []
keep_lons = np.concatenate([np.where(p_ex[0]>=st[0]-dis_lon/2)[0],np.where(p_ex[0]<=en[0]+dis_lon/2)[0]])
uni,counts_lon = np.unique(keep_lons,return_counts=True)
keep_ind.append(uni[counts_lon>=2])

keep_lats = np.concatenate([np.where(p_ex[1]>=st[1]-dis_lat/2)[0],np.where(p_ex[1]<=en[1]+dis_lat/2)[0]])
uni,counts_lat = np.unique(keep_lats,return_counts=True)
keep_ind.append(uni[counts_lat>=2])
uni,counts = np.unique(np.concatenate(keep_ind),return_counts=True)
keep_ind = uni[counts>=2]

p_ex = p_ex[:,keep_ind].transpose() #[N,2] 観測所

# points lon,lat ->  indexs i,j
inds_ex = []
for p in p_ex:
    i = np.abs(lons_df-p[0]).argmin()
    j = np.abs(lats_df-p[1]).argmin()
    inds_ex.append([i,j])
inds_ex = np.array(inds_ex).transpose()

# make mask image and sea image
mask = np.zeros([len(lats_df),len(lons_df)])
sea = np.zeros([len(lats_df),len(lons_df)])
count = 0
for ind in range(len(df)):
    # 緯度と経度の座標取得
    lonInd = int(np.where(df['lon'][ind]==lons_df)[0]) #経度
    latInd = int(np.where(df['lat'][ind]==lats_df)[0]) #緯度

    if latInd in inds_ex[1][np.where(inds_ex[0]==lonInd)[0]]: # Exist
        count += 1
        mask[latInd,lonInd] = 1
    sea[latInd,lonInd] = 1

# plot mask
plt.imshow(mask,cmap="gray")
plt.savefig(f"{out_path}/mask_eqrthquake.pdf",bbox_inches='tight')

# plot observation sites
cmap = plt.get_cmap("Greys")
cmap.set_under('grey')

mask[sea == 0] = -1

fig = plt.imshow(mask, cmap=cmap, vmin=0, interpolation="None", extent=[lon[0], lon[1], lat[0], lat[1]])
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.savefig(f"{out_path}/obs_sites.pdf",bbox_inches='tight')
#-------------------

#-------------------
# random pattern for positon map
layer = 2
size = [mask.shape[0]//(2**layer),mask.shape[1]//(2**layer)]
random_img = np.random.rand(size[0],size[1])
fig = plt.imshow(random_img, cmap="bwr", vmin=0, vmax=1, interpolation="None")
plt.savefig(f"{out_path}/random_pos.pdf",bbox_inches='tight')
pdb.set_trace()

#-------------------

