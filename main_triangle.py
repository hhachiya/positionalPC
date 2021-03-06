import pickle
import cv2
from PIL import Image
import numpy as np
import pdb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
import scipy.optimize
import scipy.stats as st
import matplotlib.pyplot as plt
import pickle
from argparse import ArgumentParser
from libs.util import PSNR
import os
from sklearn import preprocessing
from scipy.spatial import Delaunay, delaunay_plot_2d
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

def calculateThreePointAndGroup(triangle_type=1, num_nn=3):

    if triangle_type > 0:
        #--------------------
        # create triangles
        print("create triangles...")    
        tri = Delaunay(obsPoints)

        # plot
        plt.plot(targetPoints[:,1],targetPoints[:,0],'.',color="gray",markerSize=1,alpha=0.1)
        plt.triplot(obsPoints[:,1], obsPoints[:,0], tri.simplices,color="k",lw=1)
        plt.plot(obsPoints[:,1],obsPoints[:,0],'r.',markerSize=1)
        plt.gca().invert_yaxis()
        plt.savefig(triangleFile)
        #--------------------

        #--------------------
        # allocate target points to triangles
        modelingGroup = tri.simplices

        tri_inds = tri.find_simplex(targetPoints)
        groupIndexs = [list(np.where(tri_inds == i)[0]) for i in range(len(tri.simplices))]

        # allocate -1 group to nearest triangles
        no_tri_ind = np.where(tri_inds == -1)[0]
        no_tri_pnt = targetPoints[no_tri_ind]
        tri_cent = np.mean(obsPoints[tri.simplices],axis=1)

        # measure distance between pnt to centre of triangles
        no_tri_pnt_tile = np.transpose(np.tile(np.expand_dims(no_tri_pnt,0),[len(tri_cent),1,1]),[1,0,2])
        tri_cent_tile = np.tile(np.expand_dims(tri_cent,0),[len(no_tri_pnt),1,1])
        dist = np.sum(np.square(no_tri_pnt_tile - tri_cent_tile),axis=-1)
        nearest_tri_ind = np.argmin(dist,axis=1)

        # assign to nearest triangle
        [groupIndexs[nearest_tri_ind[i]].append(no_tri_ind[i]) for i in range(len(no_tri_pnt))]

        if triangle_type == 2:  # 2nd order triangle
            modelingGroup = [np.unique(np.concatenate([tri.simplices[sim_ind], np.concatenate([modelingGroup[nei_ind] if nei_ind != -1 else [] for nei_ind in tri.neighbors[sim_ind]])])).astype(int) for sim_ind in range(len(modelingGroup))]
        #--------------------
    else:
        print("find nearest obs points...")

        # compute distance of all pairs beteween obsPoints and targetPoints
        obs_num = len(obsPoints)
        target_num = len(targetPoints)
        obsPoints_tile = np.tile(np.expand_dims(obsPoints,0),[target_num,1,1])
        targetPoints_tile = np.transpose(np.tile(np.expand_dims(targetPoints,0),[obs_num,1,1]),[1,0,2])
        dist = np.sqrt(np.sum(np.square(obsPoints_tile - targetPoints_tile),axis=-1))

        # allocate to group with minimum distance
        sorted_inds = np.argsort(dist,axis=1)[:,:num_nn]
        modelingGroup = []
        groupIndexs = []
        for i in range(len(sorted_inds)):
            if i== 0:
                modelingGroup.append(sorted_inds[i])
                groupIndexs.append([i])
            else:
                tmp_dist = np.sum(modelingGroup - sorted_inds[i],axis=1)
                res=np.where(tmp_dist==0)[0]
                if len(res):
                    groupIndexs[int(res)].append(i)
                else:
                    modelingGroup.append(sorted_inds[i])
                    groupIndexs.append([i])        
   
    print("done")

    summaryData = {
        "index": groupIndexs,
        "ref": modelingGroup
    }
    pickle.dump(summaryData, open(referenceGroupFile,"wb"))

    return summaryData


def parse_args():
    parser = ArgumentParser(description="IDW & GPR using triangles")

    parser.add_argument('experiment',type=str,help='name of experiment, e.g. \'normal_PConv\'')
    parser.add_argument('-method',type=str,help='name of method, e.g. \'GPR\'')
    parser.add_argument('-dataset','--dataset',type=str,default='stripe-rectData',help='name of dataset directory (default=gaussianToyData)')
    parser.add_argument('-is_calc_group','--is_calc_group',action='store_true')
    parser.add_argument('-triangle_type','--triangle_type',type=int,default=1,help='-1: nearest neighbor, 1:triangle, 2:2nd order triangle')
    parser.add_argument('-is_2nd_triangle','--is_2nd_triangle',action='store_true')
    parser.add_argument('-num_nn','--num_nn',type=int,default=3,help='number of nearest neighbours')
    parser.add_argument('-prior_name','--prior_name',type=str,default='quake',help='prior name used for triangle.pdf and triangle.pickle')
    parser.add_argument('-posterior','--posterior',action='store_true')
    parser.add_argument('-beta',type=int,default=2, help='value of beta in IDW (default=2)')

    return parser.parse_args()

def idw(obsX, predX, y_obs, beta=1, dim=2):
    obsX = np.array(obsX)    
    obs_num = len(obsX)
    pred_num = len(predX)

    # calculate distance
    predX_tile = np.tile(np.expand_dims(predX,0), [obs_num,1,1])
    obsX_tile = np.tile(np.transpose(np.expand_dims(obsX,0),[1,0,2]),[1,pred_num,1])
    dist = np.sqrt(np.sum(np.square(predX_tile - obsX_tile),axis=-1))

    # # relative obsX w.r.t. predX
    # relX = (obsX_tile - predX_tile).transpose([1,0,2])
    
    # # sort by dist
    # sort_inds = np.argsort(dist.T,axis=1)
    # relX = np.take_along_axis(relX,np.expand_dims(sort_inds,-1),1)
    
    # # normalized dissimilality beteen relX
    # relX_norm = np.linalg.norm(relX, axis=2, keepdims=1)
    # alpha = np.matmul(relX,relX.transpose([0,2,1]))/np.matmul(relX_norm, relX_norm.transpose([0,2,1]))
    # #sin = np.where(cos >= np.cos(2*np.pi/obs_num), np.ones_like(cos), np.power(1-np.square(cos),beta/2))

    # cumulative dissimilarities
    


    # inverse distance
    weight = np.power(dist, -beta)

    # normalized weighted sum of y_obs 
    return np.dot(np.array(y_obs),weight)/np.sum(weight,axis=0)

def adjust_idw(obsX, predX, y_obs, beta=1, dim=2):
    obs_num = len(obsX)
    pred_num = len(predX)

    # calculate distance
    predX_tile = np.tile(np.expand_dims(predX,0), [obs_num,1,1])
    obsX_tile = np.tile(np.transpose(np.expand_dims(obsX,0),[1,0,2]),[1,pred_num,1])
    dist = np.sqrt(np.sum(np.square(predX_tile - obsX_tile),axis=-1))

    # inverse distance
    weight = np.power(dist, -beta)

    # normalized weighted sum of y_obs 
    return np.dot(np.array(y_obs),weight)/np.sum(weight,axis=0)

if __name__ == "__main__":
    args = parse_args()

    experiment_path = ".{0}experiment{0}{1}_logs".format(os.sep,args.experiment)
    method = args.method
    dspath = ".{0}data{0}{1}{0}".format(os.sep,args.dataset)
    if args.triangle_type==1:
        referenceGroupFile = ".{0}experiment{0}{1}_group_triangle.pickle".format(os.sep,args.prior_name)
    elif args.triangle_type==2:
        referenceGroupFile = ".{0}experiment{0}{1}_group_2nd_triangle.pickle".format(os.sep,args.prior_name)
    elif args.triangle_type==-1:
        referenceGroupFile = ".{0}experiment{0}{1}_cluster_nn.pickle".format(os.sep,args.prior_name)

    triangleFile = ".{0}experiment{0}{1}_triangle.pdf".format(os.sep,args.prior_name)
    result_path = f"{experiment_path}{os.sep}result"
    test_path = f"{result_path}{os.sep}test"

    # ???pickle??????????????????    
    TEST_PICKLE = dspath+"test.pickle"
    TEST_MASK_PICKLE = dspath+"test_mask.pickle"

    # ?????????????????????????????????
    testImg = pickle.load(open(TEST_PICKLE,"rb"))["images"]
    testLabel = pickle.load(open(TEST_PICKLE,"rb"))["labels"]
    testMask = pickle.load(open(TEST_MASK_PICKLE,"rb"))
    testMasked = testImg*testMask

    # ================
    # ??????????????????????????????
    if "stripe-rect" in args.dataset:
        # testImg = [img for img in testImg]
        testLabel = [int(lab) for lab in testLabel]
        testLabel = np.array(testLabel)
        idx = np.argsort(testLabel)
        testImg = testImg[idx]
        testLabel = testLabel[idx]
        testMasked = testMasked[idx]
    # ================

    for DIR in [experiment_path, result_path, test_path]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    mask = testMask[0,:,:,0]
    sea = np.array(Image.open(f"data{os.sep}sea.png"))/255 if ("quake" in args.dataset) else np.ones_like(mask)
    targetImg = sea - mask
    
    # ?????????:(n, d)
    obsPoints = np.array(np.where(mask > 0)).T
    obsNum = obsPoints.shape[0] 

    # ??????????????????:(m, d)
    targetPoints = np.array(np.where(targetImg > 0)).T
    targetNum = targetPoints.shape[0]

    # ?????????????????????????????????
    if args.is_calc_group:
        summaryData = calculateThreePointAndGroup(triangle_type=args.triangle_type,num_nn=args.num_nn)
    else:
        summaryData = pickle.load(open(referenceGroupFile,"rb")) 
      
    
    references = summaryData["ref"]
    groups = summaryData["index"]
    groupNum = len(groups)

    preds = []
    for predInd,img in enumerate(testImg[:,:,:,0]):
        print(f"\r {predInd} in {testImg.shape[0]}",end="")

        predImage = np.zeros([testImg.shape[1], testImg.shape[2]])
        for gind in range(groupNum):
            # ??????????????????????????????????????????????????????????????????
            group = groups[gind]

            if len(group) == 0:
                continue
            
            # 3???????????????
            threeIndex = references[gind]
            obsX = [obsPoints[i] for i in threeIndex] # ????????????????????? shape=(3, 2)
            y_obs = [ img[xy[0], xy[1]] for xy in obsX ] # ?????????????????? shape=(3)

            scaler = preprocessing.StandardScaler().fit(obsX)
            X_scaled = scaler.transform(obsX)

            # ??????????????????
            predX = [targetPoints[i] for i in group]
            predX_scaled = scaler.transform(predX)

            if method == 'GPR':
                try:
                    y_obs += np.random.random(len(y_obs))*1e-5

                    kernel = RBF()
                    #kernel = DotProduct()
                    #gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=20)
                    gp = GaussianProcessRegressor(kernel=kernel)
                    #gp = MyGPR(kernel=kernel)
                    gp.fit(obsX, y_obs)
                    #gp.fit(X_scaled, y_obs)
                    #y_mean, y_cov = gp.predict(predX_scaled, return_cov=True)
                    y_mean, y_cov = gp.predict(predX, return_cov=True)
                except:
                    pdb.set_trace()

                if args.posterior:
                    posterior = st.multivariate_normal.rvs(mean=y_mean, cov=y_cov, size=1)

            elif method == 'IDW':
                y_mean = idw(obsX, predX, y_obs, beta=args.beta)

            elif method == 'adjustIDW':
                y_mean = adjust_idw(obsX, predX, y_obs, beta=args.beta)
            
            elif method == 'OrdinaryKriging':
                obsX_array = np.array(obsX)
                predX_array = np.array(predX).astype(float)

                # avoid ill-posed problem when computing variogram
                if len(np.unique(y_obs)) == 1:
                    y_obs += np.random.random(len(y_obs))*1e-5

                OK = OrdinaryKriging(obsX_array[:,0], obsX_array[:,1], y_obs, variogram_model='linear', verbose=False, enable_plotting=False)
                z1, ss = OK.execute("grid",predX_array[:,0],predX_array[:,1])

                y_mean = np.diag(z1.data)

            elif method == 'OrdinaryKriging_exp':
                obsX_array = np.array(obsX)
                predX_array = np.array(predX).astype(float)

                # avoid ill-posed problem when computing variogram
                if len(np.unique(y_obs)) == 1:
                    y_obs += np.random.random(len(y_obs))*1e-5

                OK = OrdinaryKriging(obsX_array[:,0], obsX_array[:,1], y_obs, variogram_model='exponential', verbose=False, enable_plotting=False)
                z1, ss = OK.execute("grid",predX_array[:,0],predX_array[:,1])

                y_mean = np.diag(z1.data)

            elif method == 'OrdinaryKriging_gauss':
                obsX_array = np.array(obsX)
                predX_array = np.array(predX).astype(float)

                # avoid ill-posed problem when computing variogram
                if len(np.unique(y_obs)) == 1:
                    y_obs += np.random.random(len(y_obs))*1e-5

                OK = OrdinaryKriging(obsX_array[:,0], obsX_array[:,1], y_obs, variogram_model='exponential', verbose=False, enable_plotting=False)
                z1, ss = OK.execute("grid",predX_array[:,0],predX_array[:,1])

                y_mean = np.diag(z1.data)

            for i, p in enumerate(predX):
                predImage[p[0], p[1]] = posterior[i] if args.posterior else y_mean[i]

            for i, o in enumerate(obsX):
                predImage[o[0], o[1]] = y_obs[i]

        plt.close()
        plt.clf()
        plt.imshow(predImage)
        plt.colorbar()
        plt.savefig(f"{test_path}{os.sep}pred{predInd}.png")

        preds.append(predImage)

    pickle.dump(preds, open(f"{result_path}{os.sep}predicts.pickle","wb"))

    # ?????????????????????
    dataNum = len(preds)
    exist = sea

    errs, MAEs, MSEs, PSNRs, obsAroundPSNRs = [], [], [], [], []
    cm_bwr = plt.get_cmap("bwr") # ??????????????????
    
    site01 = mask
    ksize = 7
    dilated_site01 = cv2.dilate(site01,np.ones((ksize,ksize),np.uint8))# mask?????????
    _,dilated_site01 = cv2.threshold(dilated_site01,1e-10,1,cv2.THRESH_BINARY)# ?????????
    dilated_site01 = dilated_site01 * exist - site01 # ?????????????????????????????? 

    # ???????????????????????????
    for i, pred in enumerate(preds):
        print(f"\r progress : {i}/{dataNum}",end="")

        lab = testLabel[i]
        
        # ??????????????????
        masked = testMasked[i]
        mask = testMask[i]
        img = testImg[i]
    
        # ??????
        pred = np.squeeze(pred)
        pred[pred > 1] = 1
        pred[pred < 0] = 0

        img = np.squeeze(img)
        mask = np.squeeze(mask)
        masked = np.squeeze(masked)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ???????????????
        err = pred - img
        errs.append(err)
        # MAE
        mae = np.mean(np.abs(err))
        MAEs.append(mae)
        # MSE
        mse = np.mean(np.square(err))
        MSEs.append(mse)
        # PSNR
        psnr = PSNR(pred,img,exist=exist)
        PSNRs.append(psnr)

        # ????????????????????????????????????
        obsAroundTruth = img * dilated_site01
        obsAroundPred = pred * dilated_site01
        # ????????????????????????PSNR
        obsAroundPSNR = PSNR(obsAroundPred, obsAroundTruth, exist=dilated_site01)
        obsAroundPSNRs.append(obsAroundPSNR)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~


    # ?????????????????????
    summary_data = {
        "preds":preds,
        "error":errs,
        "MAE":np.mean(MAEs),
        "MSE":np.mean(MSEs),
        "PSNR":np.mean(PSNRs),
        "MAEs":MAEs,
        "MSEs":MSEs,
        "PSNRs":PSNRs,
        "obsAroundPSNRs":obsAroundPSNRs,
    }
    print(f"obsAround:{np.mean(obsAroundPSNRs)}")

    # ??????
    pickle.dump(summary_data, open(f"{result_path}{os.sep}testSummary.pickle","wb"))

