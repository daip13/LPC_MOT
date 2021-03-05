#!/usr/bin/env bash

set -eo pipefail
DETECTION_PATH=$1
GCN_MODEL_PATH=$2
CACHE_PATH=$3
OUTPUT_PATH=$4
GT_PATH=$5
if [[ -z ${6} ]]; then
    NUM_PROC=30
else
    NUM_PROC=$6
fi

mkdir -p ${CACHE_PATH}
rm -rf ${CACHE_PATH}/*
mkdir -p ${OUTPUT_PATH}
rm -rf ${OUTPUT_PATH}/*

starttime=`date +'%Y-%m-%d %H:%M:%S'`
SV_PATH=${CACHE_PATH}/sv/
mkdir -p ${SV_PATH}

echo "------------- Step1 data pre-processing -------------"
echo python single_view_tracker_mot.py \
    --config_path ../svonline/configs/ \
    --detection_file_path ${DETECTION_PATH} \
    --output_path ${SV_PATH} \
    --num_proc ${NUM_PROC}

python single_view_tracker_mot.py \
    --config_path ../svonline/configs/ \
    --detection_file_path ${DETECTION_PATH} \
    --output_path ${SV_PATH} \
    --num_proc ${NUM_PROC}

echo "-------------  Step2 proposal generation -------------"
PROPOSAL_PATH=${CACHE_PATH}/proposals/
GCN_DATA_PATH=${CACHE_PATH}/GCN_data/
mkdir -p ${PROPOSAL_PATH}
mkdir -p ${GCN_DATA_PATH}
echo python propsal_generation.py \
    --fe_body_track_folder ${SV_PATH} \
    --num_proc ${NUM_PROC} \
    --output_path ${PROPOSAL_PATH}

python propsal_generation.py \
    --fe_body_track_folder ${SV_PATH} \
    --num_proc ${NUM_PROC} \
    --output_path ${PROPOSAL_PATH}

echo python GCN_input_data_generation.py \
    --fe_body_track_folder ${SV_PATH} \
    --proposal_file ${PROPOSAL_PATH} \
    --output_path ${GCN_DATA_PATH}

python GCN_input_data_generation.py \
    --fe_body_track_folder ${SV_PATH} \
    --proposal_file ${PROPOSAL_PATH} \
    --output_path ${GCN_DATA_PATH}

echo "-------------  Step3 Proposal Purity Classication -------------"
INFERENCE_RESULT=${CACHE_PATH}/GCN_output/
mkdir -p ${INFERENCE_RESULT}

echo python ../gcn_based_purity_network/dsgcn/pipeline_main.py \
    --input_dir ${GCN_DATA_PATH} \
    --output_dir ${INFERENCE_RESULT} \
    --load_from1 ${GCN_MODEL_PATH}

python ../gcn_based_purity_network/dsgcn/pipeline_main.py \
    --input_dir ${GCN_DATA_PATH} \
    --output_dir ${INFERENCE_RESULT} \
    --load_from1 ${GCN_MODEL_PATH}

echo "------------- Ste4 Trajectory Inference -------------"
echo python deoverlapping.py \
    --input_path ${SV_PATH} \
    --tracklet_id_file ${GCN_DATA_PATH}/tracklet_id_transfer.json \
    --proposal_path ${GCN_DATA_PATH}/eval1/ \
    --GCN_output_file ${INFERENCE_RESULT}/Estimated_IoP_eval.json \
    --num_proc ${NUM_PROC} \
    --output_path ${OUTPUT_PATH} 

python deoverlapping.py \
    --input_path ${SV_PATH} \
    --tracklet_id_file ${GCN_DATA_PATH}/tracklet_id_transfer.json \
    --proposal_path ${GCN_DATA_PATH}/eval1/ \
    --GCN_output_file ${INFERENCE_RESULT}/Estimated_IoP_eval.json \
    --num_proc ${NUM_PROC} \
    --output_path ${OUTPUT_PATH} 

echo "------------- Step5 Post-processing -------------"
echo python post_processing.py \
    --input_path ${OUTPUT_PATH} \
    --output_path ${OUTPUT_PATH}

python post_processing.py \
    --input_path ${OUTPUT_PATH} \
    --output_path ${OUTPUT_PATH}

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s)
end_seconds=$(date --date="$endtime" +%s)
echo "The processing time for LPC_MOT with ${NUM_PROC} number of processors is: "$((end_seconds-start_seconds))"s"

echo "------------- Evaluation -------------"
echo python mot_metric_evaluation.py \
    --out_mot_files_path ${OUTPUT_PATH} \
    --gt_path ${GT_PATH} 

python mot_metric_evaluation.py \
    --out_mot_files_path ${OUTPUT_PATH} \
    --gt_path ${GT_PATH} |& tee ${OUTPUT_PATH}/eval.log
