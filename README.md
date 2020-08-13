# DERT

###Multi-Context Attention Network for Diversity of Entity Resolution Tasks

***
## Background
The code is the open source code of the MCAN model proposed to solve multiple entity analysis tasks, the comparison algorithm code and the data set required for testing. The data set needs to be obtained through the following methods for some reasons.
***
## Install
Code operating environment

### Text-ER
The model running environment is in ./text_er/requirements.txt. Only need
`pip install -r requirments.txt`
to establish the running environment corresponding to Video-ER tasks.

The [word embedding](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip) used in the fasttext word vector model needs to be downloaded to the ./text_er/dataset/ directory in advance.


### Video-ER
The model running environment is in ./video_er/requirements.txt. Only need
`pip install -r requirments.txt`
to establish the running environment corresponding to Video-ER tasks.

### Traj-ER
Due to the complexity of the Traj-ER task code, the environment requires'requirements.txt' to be placed in the corresponding algorithm directory.

### Check-in-ER
Due to the complexity of the Check-in-ER task code, the environment requires'requirements.txt' to be placed in the corresponding algorithm directory.

***

## Dataset 

### Video-ER

#####[Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)
#####[Occluded-DukeMTMC](https://github.com/lightas/Occluded-DukeMTMC-Dataset)

The Video-ER task data set is Market1501 and Occluded-DukeMTMC. After downloading the data, unzip the Market1501 data set to [./video_er/market1501](https://github.com/lzzppp/DERT/tree/master/video_er/market1501)/bounding_box_train and [./video_er/market1501](https://github.com/lzzppp/DERT/tree/master/video_er/market1501)/bounding_box_test unzip the Occluded_dukeMTMC data set to [./video_er/occluded_dukeMTMC](https://github.com/lzzppp/DERT/tree/master/video_er/occluded_dukeMTMC)/bounding_box_train and [./video_er/occluded_dukeMTMC](https://github.com/lzzppp/DERT/tree/master/video_er/occluded_dukeMTMC)/bounding_box_test

The Video-ER data is in the form of pair, and the data sets for train, valid, and test are stored in the [./video_er/market1501/](https://github.com/lzzppp/DERT/tree/master/video_er/market1501) and [./video_er/occluded_dukeMTMC](https://github.com/lzzppp/DERT/tree/master/video_er/occluded_dukeMTMC) directories.The files are named train_list.csv, valid_list.csv, test_list.csv.

### Traj-ER

####source_data
+ #####[Geolife](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
+ #####[Singapore](https://www.microsoft.com/en-us/download/details.aspx?id=52367)

####Processed data
After filtering the source data for dirty data and removing too short data, the data is used to generate train test data used by all comparison algorithms of the Traj-ER task.

+ #####[traj_dataset](https://drive.google.com/drive/folders/18l9jCdGc0J7Z6mZ3FtxPYX56Qa9MmNrj?usp=sharing)
Download the Singapore and Geolife MCAN algorithm and comparison algorithm data sets of the Traj-ER task from Google Cloud Disk to the [./traj_er/dataset/](https://github.com/lzzppp/DERT/tree/master/traj_er/dataset) directory.

### Check-In-ER

+ ##### [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
+ ##### [Gowalla](http://snap.stanford.edu/data/loc-gowalla.html)

####Processed data
After filtering the source data for dirty data and removing too short data, the data is used to generate train test data used by all comparison algorithms of the Check-in-ER task.

+ #####[check_in_dataset](https://drive.google.com/drive/folders/1XWAAsjdNJ4lUsuEMX8QawSGqUKMe96_z?usp=sharing)
Download the Foursquare and Gowalla MCAN algorithm and comparison algorithm data sets of the Check-in-ER task from Google Cloud Disk to the [./check_in_er/dataset/](https://github.com/lzzppp/DERT/tree/master/check_in_er/dataset) directory.

***

## Code operation

### Text-ER
#### data preparation
#####[Dataset](http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)

The data set is downloaded and stored in the ./text_er/dataset/ directory.
Since our model uses the vector of the pre-trained fasttext model, we need to download the [wiki.en.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip) file in advance to the [./text_er/MCAN/dataset/](https://github.com/lzzppp/DERT/tree/master/text_er/MCAN/dataset) directory.

####Training model
After downloading the data set and pre-training vector file, enter the virtual environment under the Text-ER task. Then go to the ./text_er/MCAN/ directory and run the following command.
```
  python main.py
--type TYPE
--dataset DATASET
--batch_size BATCH_SIZE
--epoch EPOCH
--pos_neg POS_NEG
--ablation_part ABLATION_PART
```
The TYPE parameter indicates the form of the data set, including [StructuralWithValue, Structural, Textual, Dirty, Dirty1, Dirty2] and other forms. The DATASET parameter represents the name of the training data set, specifically the name of the data set folder, depending on the situation. The BATCH_SIZE parameter is the training batch, the default is 16, and it is recommended to remain unchanged. The EPOCH parameter is the number of training rounds, and the default is 10. The POS_NEG parameter indicates whether the model training considers the problem of the ratio of positive and negative samples. The ABLATION_PART parameter is the parameter required for ablation experiment. String input such as SA, PA, GM, GA can remove a part of the model for comparison experiment. The default is'none'.

### Video-ER

#### Convenient training
After downloading the dataset Market1501 and Occluded-DukeMTMC of the Video-ER task, if you want to train all models including [MCAN, RGA-Net, RGA-Net-PA, ABD-Net], you can directly run `bash ./video_er/ run.sh` will directly perform the training process of all models on all data sets.
#### Train a model individually
After downloading the data set Market1501 and Occluded-DukeMTMC of the Video-ER task, if you want to train all models [MCAN, RGA-Net, RGA-Net-PA, ABD-Net], one of the models in the existing data set Individual training on [market1501, occluded_dukeMTMC]. First enter the [./video_er/video_er_models/](https://github.com/lzzppp/DERT/tree/master/video_er/video_er_models) folder directory and run the command
```
  python main.py
--model_type MODEL 
--dataset DATASET 
--epoch EPOCH
--Warm-up-epoch WU-EPOCH
--pos_neg POS_NEG
```
The MODEL parameter is to determine the model to be trained, and select a model to be trained from the list [MCAN, RGA-Net, RGA-Net-PA, ABD-Net]. The DATASET parameter indicates the data set to be trained. Choose a data set to be trained from the list [market1501, occluded_dukeMTMC]. The EPOCH parameter indicates the number of rounds of model training. The default is 40. Increasing the number of training rounds may improve the model results, but there will be a long training time.
WU-EPOCH is the number of warm-up training rounds, that is, it controls the number of convolutional network layers and only trains the number of training rounds for the network parameters behind the model. Large adjustments to this parameter may improve some models such as MCAN, which is also the advantage of the MCAN model. The POS_NEG parameter indicates whether to consider the positive and negative sample ratio of the training data during model training. It is considered by default, which is conducive to more stable model training.

####Additional models and additional data sets
If we want to make minor modifications to the model or use a better convolutional layer to improve the model level, we can add the model in the [./video_er/video_er_models](https://github.com/lzzppp/DERT/tree/master/video_er/video_er_models) directory and modify the [set_models.py](https://github.com/lzzppp/DERT/blob/master/video_er/video_er_models/set_models.py) file to add the new model to the trainable In sequence.
Experimenters can expand or change the training data set to explore new possibilities to improve the performance of the model. They only need to organize the data set into a format such as market1501 and add it to the training data list to use the new dataset.

