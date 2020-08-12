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
#####[Geolife](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
#####[Singapore](https://www.microsoft.com/en-us/download/details.aspx?id=52367)

####Processed data
After filtering the source data for dirty data and removing too short data, the data is used to generate training test data used by all comparison algorithms of the Traj-ER task.

#####[traj_dataset](https://drive.google.com/drive/folders/18l9jCdGc0J7Z6mZ3FtxPYX56Qa9MmNrj?usp=sharing)
Download the Singapore and Geolife MCAN algorithm and comparison algorithm data sets of the Traj-ER task from Google Cloud Disk to the [./traj_er/dataset/](https://github.com/lzzppp/DERT/tree/master/traj_er/dataset) directory.

### Check-In-ER

#####[Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
#####[Gowalla](http://snap.stanford.edu/data/loc-gowalla.html)

