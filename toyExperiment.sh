#!/bin/bash

# 学習とテストを行うシェルスクリプトファイル

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# toy data
# v2 は U-Net、siteBranchを1Epochずつ学習
# v3 は U-Netを4Epoch学習後、siteBranchを1Epoch学習

#center="stripe-rectData256_v3"
center="stripe-rectData256_v2"
#center="stripe-rectData"
dspath="-dataset ${center}"
size="-imgw 256 -imgh 256"
small_lr=0.001
is_nonNeg=0

########################################################
# 既存法 (IDW)
# method="IDW"
# for beta in 1 2
# do
#     exp="${method}_beta${beta}_${center}"
#     if [ $beta -eq 1 ]; then
#         python main_triangle.py ${exp} ${dspath} -method ${method} -beta ${beta} -prior_name=${center} -is_calc_group
#     else
#         python main_triangle.py ${exp} ${dspath} -method ${method} -beta ${beta} -prior_name=${center}
#     fi
# done

# # 既存法 (Kriging)
# method="OrdinaryKriging"
# exp="${method}_${center}"
# python main_triangle.py ${exp} ${dspath} -method ${method} -prior_name=${center}

# 既存法 (GPR)
# method="GPR"
# exp="${method}_${center}"
# python main_triangle.py ${exp} ${dspath} -method ${method} -prior_name=${center}

# 既存法 (pconv)
exp="pconv_${center}"
python main.py ${exp} ${dspath} ${size}
python test.py ${exp} ${dspath} ${size}
########################################################        

########################################################
# Branchあり
## U-Netを学習後に位置特性のCNNを学習

#for siteLayer in 2 3 4
for siteLayer in 2 3 4
do
    # # phase = 1
    if [ ${is_nonNeg} -eq 1 ]; then
        pretrainExp="ph1_pkconv_${center}_branchlsiteF-nonNeg-l${siteLayer}"
        modelconfig="-branchlSitePKConv -siteLayers ${siteLayer} -nonNegSConv"
    else
        pretrainExp="ph1_pkconv_${center}_branchlsiteF-l${siteLayer}"
        modelconfig="-branchlSitePKConv -siteLayers ${siteLayer}"
    fi    
    python main.py ${pretrainExp} ${dspath} ${modelconfig} ${size} -phase 1
    python test.py ${pretrainExp} ${dspath} ${modelconfig} ${size} -phase 1 -pt ${pretrainExp}

    # 乗算
    # phase = 2 -> 3
    if [ ${is_nonNeg} -eq 1 ]; then
        exp="ph2-3_pkconv_${center}_mul_branchlsiteF-nonNeg-l${siteLayer}"
        modelconfig="-ope mul -branchlSitePKConv -siteLayers ${siteLayer} -nonNegSConv"
    else
        exp="ph2-3_pkconv_${center}_mul_branchlsiteF-l${siteLayer}"
        modelconfig="-ope mul -branchlSitePKConv -siteLayers ${siteLayer}"
    fi
    python main.py ${exp} ${dspath} ${modelconfig} ${size} -phase 2 -pt ${pretrainExp} -lr ${small_lr}
    python main.py ${exp} ${dspath} ${modelconfig} ${size} -phase 3 -pt ${exp} -lr ${small_lr}
    python test.py ${exp} ${dspath} ${modelconfig} ${size} -phase 2
    python test.py ${exp} ${dspath} ${modelconfig} ${size} -phase 3

    # 加算1
    # phase = 2 -> 3
    exp="ph2-3_pkconv_${center}_add1_branchlsiteF-nonNeg-l${siteLayer}"
    modelconfig="-ope addX -branchlSitePKConv -siteLayers ${siteLayer} -nonNegSConv"
    # python main.py ${exp} ${dspath} ${modelconfig} -phase 2 -pt ${pretrainExp} -lr ${small_lr}
    # python main.py ${exp} ${dspath} ${modelconfig} -phase 3 -pt ${exp} -lr ${small_lr}
    # python test.py ${exp} ${dspath} ${modelconfig} -phase 2
    # python test.py ${exp} ${dspath} ${modelconfig} -phase 3

    # 加算2
    # phase = 2 -> 3
    exp="ph2-3_pkconv_${center}_add2_branchlsiteF-nonNeg-l${siteLayer}"
    modelconfig="-ope add -branchlSitePKConv -siteLayers ${siteLayer} -nonNegSConv"
    # python main.py ${exp} ${dspath} ${modelconfig} -phase 2 -pt ${pretrainExp} -lr ${small_lr}
    # python main.py ${exp} ${dspath} ${modelconfig} -phase 3 -pt ${exp} -lr ${small_lr}
    # python test.py ${exp} ${dspath} ${modelconfig} -phase 2
    # python test.py ${exp} ${dspath} ${modelconfig} -phase 3

done

# python libs/compare_exp.py "実験に使ったデータフォルダの名前" "experimentsに保存されている実験の名前" -comp -region -spath "画像を保存するフォルダの名前"

