#!/bin/bash


# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64

#create folder if not exist
#error wenn ordnerstruktur nicht der konvention entspricht
#auslesen der Bezeichnungen anhand der ordnernamen

slice="Leervermessung"
cam="KreuzungDomplatz"
day="20180318"
pathToFirstHd="media/ecl/data3"
pathToSecondHd="media/ecl/6448CDFF1E7ADFEE" 
pathToDetectron="home/ecl/detectron"
pathToDeepSort="media/ecl/DATA1/deep_sort"
numPartsMax=2
numSubPartsMax=8

for j in $(seq -f "%02g" 1 $numPartsMax); do 
        
    for i in $(seq -f "%02g" 1 $numSubPartsMax); do

		echo "/$pathToFirstHd/$slice/$cam/$day/det/$j/detections_with_features_$i.npy"

		export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64

        python "/$pathToDetectron/tools/infer_simple.py" \
        --cfg "/$pathToDetectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml" \
        --output-dir "/$pathToFirstHd/$slice/$cam/$day/results_maskrcnn_25fps/$j" \
		--outputdirdet "/$pathToFirstHd/$slice/$cam/$day/det/$j" \
		--sub-part "$i" \
        --image-ext jpg  \
        --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
		"/$pathToSecondHd/$slice/$cam/$day/img1/$j"

		export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64

        python "/$pathToDeepSort/tools/generate_detections.py" \
        --model "/$pathToDeepSort/resources/networks/mars-small128.pb" \
        --mot_dir "/$pathToSecondHd/$slice/$cam/$day" \
		--detection_dir "/$pathToFirstHd/$slice/$cam/$day" \
		--part "$j" \
        --subpart "$i" \
        --output_dir "/$pathToFirstHd/$slice/$cam/$day/det/$j"

        python "/$pathToDeepSort/deep_sort_app.py" \
        --sequence_dir "/$pathToSecondHd/$slice/$cam/$day" \
        --detection_file "/$pathToFirstHd/$slice/$cam/$day/det/$j/detections_with_features_$i.npy" \
        --output_file "/$pathToFirstHd/$slice/$cam/$day/tracks/$j/tracks_$i.txt" \
        --part "$j" \
        --subpart "$i" \
        --min_confidence 0.7 \
        --nn_budget 100 \
        --display True
    done
done
