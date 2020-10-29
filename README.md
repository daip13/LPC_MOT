# LPC_MOT
This is the code for the paper "Learning a Proposal Classifier for Multiple Target tracking"
![image](https://github.com/daip13/LPC_MOT/blob/master/images/framework.png)

## Setup
1. Clone the enter this repository:
```
git clone https://github.com/daip13/LPC_MOT.git
```

2. Create a docker image for this project: 
    - Python = 3.7.7
    - PyTorch = 1.4.0+cu100
    - Notice: We also provide the docker image [Baidu](https://pan.baidu.com/s/1IF7JqycSzP6iqbR9fkduJA) (code: lq3v) to run our codes.

3. Copy the LPC_MOT repository to the root path of the docker image.

4. Download our GCN and reid network.
    - The models can also be downloaded [Baidu](https://pan.baidu.com/s/1IF7JqycSzP6iqbR9fkduJA) (code: lq3v).
    - You should place the models to path /root/LPC_MOT/models/
    - Notice: we adopt the [fast-reid](https://github.com/JDAI-CV/fast-reid.git) as our reid model. However, the authors have updated their codes. In order to get the same reid features with our trained model, we also present the codes that we used here.

5. (OPTIONAL) For convenience, we provide the detections files with extracted reid features. You can also download them [Baidu](https://pan.baidu.com/s/1IF7JqycSzP6iqbR9fkduJA) (code: lq3v).
    - You should place the downloaded data to /root/LPC_MOT/dataset/
    - If you donot want to download the data, you can also generate it with the script [ReID_feature_extraction.py](https://github.com/daip13/LPC_MOT/blob/master/learnable_proposal_classifier/scripts/ReID_feature_extraction.py)

6. Download the MOT17 dataset and place it to path /root/LPC_MOT/dataset/.

7. Running.
```
cd /root/LPC_MOT/learnable_proposal_classifier/scripts/
bash main.sh ../../dataset/MOT17/results_reid_with_traindata/detection/ ../../models/dsgcn_model_iter_30.pth /tmp/LPC_MHT/ ../../dataset/MOT17/results_reid_with_traindata/tracking_output/ ../../dataset/MOT17/train/
```

## GCN Model Training
The scripts for GCN model training will be here soon.
