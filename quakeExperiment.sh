#!/bin/bash

# 学習とテストを行うシェルスクリプトファイル

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 地震動データ
# v2 は U-Net、siteBranchを1Epochずつ学習
# v3 は U-Netを4Epoch学習後、siteBranchを1Epoch学習
small_lr=0.001
is_nonNeg=0
is_calc_group=1
#for i in `seq 0 4`
for i in `seq 0 9`
do
    #for center in "all" "h01h02h06h07" "h04h05h09h10"    
    for center in "all"
    do
        dspath="-dataset quakeData-${center}-crossVaridation${i}"

        for triangle_type in 2 1
        do
            ########################################################
            # 既存法 (IDW)
            method="IDW"            
            for beta in 1 2
            do
                exp="${method}_beta${beta}_triangle${triangle_type}_${center}_cv${i}"
                #if [ $i -eq 0 ] && [ $beta -eq 1 ]; then
                if [ $is_calc_group -eq 1 ]; then
                    python main_triangle.py ${exp} ${dspath} -method ${method} -beta ${beta} -prior_name=quake -is_calc_group -triangle_type=${triangle_type}
                    is_calc_group=0
                else
                    python main_triangle.py ${exp} ${dspath} -method ${method} -beta ${beta} -prior_name=quake -triangle_type=${triangle_type}
                fi
            done

            # 既存法 (Kriging with linear variogram)
            method="OrdinaryKriging"
            exp="${method}_triangle${triangle_type}_${center}_cv${i}"
            python main_triangle.py ${exp} ${dspath} -method ${method} -prior_name=quake -triangle_type=${triangle_type}

            # 既存法 (Kriging with gaussian variogram)
            method="OrdinaryKriging_gauss"
            # exp="${method}_triangle${triangle_type}_${center}_cv${i}"
            # python main_triangle.py ${exp} ${dspath} -method ${method} -prior_name=quake -triangle_type=${triangle_type}

            # 既存法 (Kriging with exponential variogram)
            method="OrdinaryKriging_exp"
            exp="${method}_triangle${triangle_type}_${center}_cv${i}"
            python main_triangle.py ${exp} ${dspath} -method ${method} -prior_name=quake -triangle_type=${triangle_type}
        done       

        #既存法 (GPR)
        method="GPR"
        # exp="${method}_${center}_cv${i}"
        # python main_triangle.py ${exp} ${dspath} -method ${method} -prior_name=quake -triangle_type=2

        # 既存法 (pconv)
        # exp="pconv_${center}_cv${i}"
        # python main.py ${exp} ${dspath} -existOnly
        # python test.py ${exp} ${dspath}
        ########################################################        

        ########################################################
        # Branchあり
        ## U-Netを学習後に位置特性のCNNを学習

        #for siteLayer in 2 3 4
        for siteLayer in 2
        do
            # phase = 1
            if [ ${is_nonNeg} -eq 1 ]; then
                pretrainExp="ph1_pkconv_${center}_branchlsiteF-nonNeg-l${siteLayer}_cv${i}"
                modelconfig="-branchlSitePKConv -siteLayers ${siteLayer} -nonNegSConv"
            else
                pretrainExp="ph1_pkconv_${center}_branchlsiteF-l${siteLayer}_cv${i}"
                modelconfig="-branchlSitePKConv -siteLayers ${siteLayer}"
            fi
            # python main.py ${pretrainExp} ${dspath} ${modelconfig} -existOnly -phase 1
            # python test.py ${pretrainExp} ${dspath} ${modelconfig} -phase 1 -pt ${pretrainExp}

            # 乗算
            # phase = 2 -> 3
            if [ ${is_nonNeg} -eq 1 ]; then
                exp="ph2-3_pkconv_${center}_mul_branchlsiteF-nonNeg-l${siteLayer}_cv${i}"
                modelconfig="-ope mul -branchlSitePKConv -siteLayers ${siteLayer} -nonNegSConv"
            else
                exp="ph2-3_pkconv_${center}_mul_branchlsiteF-l${siteLayer}_cv${i}"
                modelconfig="-ope mul -branchlSitePKConv -siteLayers ${siteLayer}"
            fi
            # python main.py ${exp} ${dspath} ${modelconfig} -existOnly -phase 2 -pt ${pretrainExp} -lr ${small_lr}
            # python main.py ${exp} ${dspath} ${modelconfig} -existOnly -phase 3 -pt ${exp} -lr ${small_lr}
            # python test.py ${exp} ${dspath} ${modelconfig} -phase 2
            # python test.py ${exp} ${dspath} ${modelconfig} -phase 3


            # 加算2
            # phase = 2 -> 3
            if [ ${is_nonNeg} -eq 1 ]; then
                exp="ph2-3_pkconv_${center}_add2_branchlsiteF-nonNeg-l${siteLayer}_cv${i}"
                modelconfig="-ope add -branchlSitePKConv -siteLayers ${siteLayer} -nonNegSConv"
            else
                exp="ph2-3_pkconv_${center}_add2_branchlsiteF-l${siteLayer}_cv${i}"
                modelconfig="-ope add -branchlSitePKConv -siteLayers ${siteLayer}"
            fi            
            # python main.py ${exp} ${dspath} ${modelconfig} -existOnly -phase 2 -pt ${pretrainExp} -lr ${small_lr}
            # python main.py ${exp} ${dspath} ${modelconfig} -existOnly -phase 3 -pt ${exp} -lr ${small_lr}
            # python test.py ${exp} ${dspath} ${modelconfig} -phase 2
            # python test.py ${exp} ${dspath} ${modelconfig} -phase 3

            # affine
            # phase = 2 -> 3
            if [ ${is_nonNeg} -eq 1 ]; then
                exp="ph2-3_pkconv_${center}_affine_branchlsiteF-nonNeg-l${siteLayer}_cv${i}"
                modelconfig="-ope affine -branchlSitePKConv -siteLayers ${siteLayer} -nonNegSConv"
            else
                exp="ph2-3_pkconv_${center}_affine_branchlsiteF-l${siteLayer}_cv${i}"
                modelconfig="-ope affine -branchlSitePKConv -siteLayers ${siteLayer}"
            fi            
            # python main.py ${exp} ${dspath} ${modelconfig} -existOnly -phase 2 -pt ${pretrainExp} -lr ${small_lr}
            # python main.py ${exp} ${dspath} ${modelconfig} -existOnly -phase 3 -pt ${exp} -lr ${small_lr}
            # python test.py ${exp} ${dspath} ${modelconfig} -phase 2
            # python test.py ${exp} ${dspath} ${modelconfig} -phase 3

        done
    done
done

# python libs/compare_exp.py "実験に使ったデータフォルダの名前" "experimentsに保存されている実験の名前" -comp -region -spath "画像を保存するフォルダの名前"

