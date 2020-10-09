# BMN: Boundary-Matching Network for Temporal Action Proposal Generation   

### 1. Introduction  

This is my implementation of [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702)(2019 ICCV) in pure Pytorch. It contains the full pipeline of training and evaluation on ActivityNet-1.3. The key features of this repo are:

- Full training and evaluation pipeline.
- Automatic parameter saving, loading and changing through command line.  

### 2. Requirements

Python version: 2 or 3

Packages:

- Pytorch >= 1.0
- pandas
- tqdm

### 3. Accuracy

After 9 epochs, we can get accuracy as below:

| AN     | Recall |
| ------ | ------ |
| AR@1   | 33.6%  |
| AR@5   | 49.7%  |
| AR@10  | 57.0%  |
| AR@100 | 75.4%  |
| AUC    | 67.4%  |  

### 4. Tutorial

#### 4.3 How to use the computer

登录到电脑之后，东西都在`/media/e813/D/wzt/`这个文件夹下面，具体的代码一般在`./codes`文件夹下面。  
这个仓库在`./codes/Pytorch-BMN`里面，可以再建一个文件夹，然后新建一个`branch`来放自己的代码。

#### 4.2 data prepare

All data annotations are saved in folder 'data', specific list is as follows:  

data   

-- activitynet_annotations   
-- · action_name.csv  
-- · anet_anno_action.json  
-- · video_info_new.csv   

-- eval  
-- · activity_net_1_3_new.json  
-- · activity_net.v1-3.min.json  
-- · sample_classification_prediction.json  
-- · sample_detection_prediction.json  

**我们自己的数据**  

我们使用ActivityNet作为数据集，因为这个数据集有已经提取好的特征，我们的算力不太支持我们从提取特征开始做。我们可能会用到原视频，在实验室也有，但是因为太大了，暂时没有转存到这个电脑上。

We use same extracted-feature and data as [BSN](https://github.com/wzmsltw/BSN-boundary-sensitive-network).  

提取好的特征的路径: 

/media/e813/D/wzt/datasets/Activitynet/   

所有的提取好的特征已经下载下来到上面的文件夹里面了。每一个视频里面对应一个csv文件，可以使用pandas进行读取。  
长度限制，每个视频的被提取成了长度100，维度400的特征向量。


We support experiments with publicly available dataset ActivityNet 1.3 for temporal action proposal generation now. To download this dataset, please use [official ActivityNet downloader](https://github.com/activitynet/ActivityNet/tree/master/Crawler) to download videos from the YouTube.

To extract visual feature, we adopt TSN model pretrained on the training set of ActivityNet, which is the challenge solution of CUHK&ETH&SIAT team in ActivityNet challenge 2016. Please refer this repo [TSN-yjxiong](https://github.com/yjxiong/temporal-segment-networks) to extract frames and optical flow and refer this repo [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk) to find pretrained TSN model.

For convenience of training and testing, we rescale the feature length of all videos to same length 100, and we provide the rescaled feature at here [Google Cloud](https://drive.google.com/file/d/1ISemndlSDS2FtqQOKL0t3Cjj9yk2yznF/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/19GI3_-uZbd_XynUO6g-8YQ). If you download features from BaiduYun, please use `cat zip_csv_mean_100.z* > csv_mean_100.zip` before unzip. After download and unzip, please put `csv_mean_100` directory to `./data/activitynet_feature_cuhk/` .

#### 4.3 train

Using `train.py` for training. Check the `opt.py` for more details. You may set the parameters yourself in your own specific task.

```shell
python train.py -- epochs 30
```
 
#### 4.4 test

Same as above, using `test.py` for testing.

```Shell
python test.py
```

### 5. Reference

I referred to many fantastic repos during the implementation:  

[Tianwei Lin-BMN-Paddle](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/video/models/bmn)       
[JJBOY-BMN-Pytorch](https://github.com/JJBOY/BMN-Boundary-Matching-Network)    


### 6. TODO
[ ] mAP calculating.  


