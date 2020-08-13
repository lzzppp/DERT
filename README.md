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

+ ##### [Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)
+ ##### [Occluded-DukeMTMC](https://github.com/lightas/Occluded-DukeMTMC-Dataset)

The Video-ER task data set is Market1501 and Occluded-DukeMTMC. After downloading the data, unzip the Market1501 data set to [./video_er/market1501](https://github.com/lzzppp/DERT/tree/master/video_er/market1501)/bounding_box_train and [./video_er/market1501](https://github.com/lzzppp/DERT/tree/master/video_er/market1501)/bounding_box_test unzip the Occluded_dukeMTMC data set to [./video_er/occluded_dukeMTMC](https://github.com/lzzppp/DERT/tree/master/video_er/occluded_dukeMTMC)/bounding_box_train and [./video_er/occluded_dukeMTMC](https://github.com/lzzppp/DERT/tree/master/video_er/occluded_dukeMTMC)/bounding_box_test

The Video-ER data is in the form of pair, and the data sets for train, valid, and test are stored in the [./video_er/market1501/](https://github.com/lzzppp/DERT/tree/master/video_er/market1501) and [./video_er/occluded_dukeMTMC](https://github.com/lzzppp/DERT/tree/master/video_er/occluded_dukeMTMC) directories.The files are named train_list.csv, valid_list.csv, test_list.csv.

### Traj-ER

#### source_data
+ #####[Geolife](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
+ #####[Singapore](https://www.microsoft.com/en-us/download/details.aspx?id=52367)

#### Processed data
After filtering the source data for dirty data and removing too short data, the data is used to generate train test data used by all comparison algorithms of the Traj-ER task.

+ ##### [traj_dataset](https://drive.google.com/drive/folders/18l9jCdGc0J7Z6mZ3FtxPYX56Qa9MmNrj?usp=sharing)
Download the Singapore and Geolife MCAN algorithm and comparison algorithm data sets of the Traj-ER task from Google Cloud Disk to the [./traj_er/dataset/](https://github.com/lzzppp/DERT/tree/master/traj_er/dataset) directory.

### Check-In-ER

+ ##### [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
+ ##### [Gowalla](http://snap.stanford.edu/data/loc-gowalla.html)

#### Processed data
After filtering the source data for dirty data and removing too short data, the data is used to generate train test data used by all comparison algorithms of the Check-in-ER task.

+ ##### [check_in_dataset](https://drive.google.com/drive/folders/1XWAAsjdNJ4lUsuEMX8QawSGqUKMe96_z?usp=sharing)
Download the Foursquare and Gowalla MCAN algorithm and comparison algorithm data sets of the Check-in-ER task from Google Cloud Disk to the [./check_in_er/dataset/](https://github.com/lzzppp/DERT/tree/master/check_in_er/dataset) directory.

***

## Code operation

### Text-ER
#### data preparation
##### [Dataset](http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)

The data set is downloaded and stored in the ./text_er/dataset/ directory.
Since our model uses the vector of the pre-trained fasttext model, we need to download the [wiki.en.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip) file in advance to the [./text_er/MCAN/dataset/](https://github.com/lzzppp/DERT/tree/master/text_er/MCAN/dataset) directory.

#### Training model
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
The TYPE parameter indicates the form of the data set, including **[StructuralWithValue, Structural, Textual, Dirty, Dirty1, Dirty2]** and other forms. The DATASET parameter represents the name of the training data set, specifically the name of the data set folder, depending on the situation. The BATCH_SIZE parameter is the training batch, the default is 16, and it is recommended to remain unchanged. The EPOCH parameter is the number of training rounds, and the default is 10. The POS_NEG parameter indicates whether the model training considers the problem of the ratio of positive and negative samples. The ABLATION_PART parameter is the parameter required for ablation experiment. String input such as **SA, PA, GM, GA** can remove a part of the model for comparison experiment. The default is'none'.

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

#### Additional models and additional data sets
If we want to make minor modifications to the model or use a better convolutional layer to improve the model level, we can add the model in the [./video_er/video_er_models](https://github.com/lzzppp/DERT/tree/master/video_er/video_er_models) directory and modify the [set_models.py](https://github.com/lzzppp/DERT/blob/master/video_er/video_er_models/set_models.py) file to add the new model to the trainable In sequence.
Experimenters can expand or change the training data set to explore new possibilities to improve the performance of the model. They only need to organize the data set into a format such as market1501 and add it to the training data list to use the new dataset.

### Traj-ER

#### MCAN and comparison algorithm

#### MCAN

Code path 10090 port server /home/zhuzheng/papercode/mcan/

The data path on the server is: /mnt/disk_data_hdd/zhuzheng/trajectory/mcan

/home/zhuzheng/papercode/mcan/dataset is the original corpus data, it is best not to move

Environment: /home/nieyuyang/py3 (virtualenv)

The program entry is main2.py, which itself is a code that handles entity resolution in nlp. Currently, it uses trajectory cell sequence input instead.

#### main.py

There are three parameters to note:

-dataset: the input data set

-length: trajectory length, corresponding to the length of to_mcan_format.py in data preprocessing

-ablation_part: The structure of each attention (SA PA GM GA) is detailed in the paper



In addition, because this code is directly modified from the nlp code, a large amount of the data preprocessing part of the code is the original nlp code, which is more complicated. It is recommended to modify the model part first, and then modify and integrate the preprocessing part before open source. This nlp code will involve loading pre-trained embedding (larger), this part of which I did not change in order to run quickly, so it is best to keep the dataset folder unchanged when running.

#### Attention visualization

Mainly to visualize the value of Global Attention.

The last three lines in Main2.py will store the attention in the data file directory of the corresponding dataset. The analyze_gb_attention.py mainly calculates the average rank value of the attentioin size of the cell where the stay point is located.

analysis/cell_vis.py is used to visualize the size of each cell attention value, which is also used in jupyter notebook

#### DPLink

Code path 10090 port server [./traj_er/DPLink/codes/](https://github.com/lzzppp/DERT/tree/master/traj_er/DPLink/codes)

Environment: python2 [./traj_er/DPLink/requirments.txt](https://github.com/lzzppp/DERT/tree/master/traj_er/DPLink)

The main entrance is run_my_data.py

The first parameter that needs attention is vid_size, which is printed out in the preprocessing file. See the preprocessing document for details. That value needs to be set in the code. Currently, I am running data of different data sets and different lengths of time. The specific settings are in lines 29-33 of run_my_data.py:

```python
vid_size_dict = {
    "singapore":{24.0 : 13613},
    "geolife": {24.0 : 57754}
}
```

Fill in the data according to the corresponding length of the corresponding data set.

The model-related parameters are mainly in the settings of lines 73-115.

The parameters of the parser need to be adjusted by yourself: --data, --gpu --epoch --length --cell_width and other defaults.

Among them, cell_width is that I have tried different cell width divisions, and need to fill in the preprocessed width (corresponding to to_DPlink_format.py in preprocessing)

#### T2vec

Code path: [./traj_er/t2vec_experience/requirments.txt](https://github.com/lzzppp/DERT/tree/master/traj_er/t2vec_experience)

The purpose of this comparison algorithm itself is to input a trajectory and output the embedding of this trajectory. The way we use it as the comparison algorithm is to splice the embeddings of each pair of trajectories and use a neural network to judge whether it comes from the same person.

The process of running the entire algorithm is cumbersome. I have written two bash files: run_part1.sh run_part2.sh

1. Open the hyper-parameters.json file, refer to other parameter formats, and fill in the specific settings:

   ```json
   "region_singapore_len24.0":
     {
       "cityname": "singapore_pair24.0",
       "minlon":103.5997,
       "minlat":1.2077,
       "maxlon":104.0907,
       "maxlat":1.4809,
       "cellsize":100.0,
       "minfreq":100,
       "filepath":"...",
       "testpath":"...",
       "train_num":7129,
       "vocab_size":6878
     }
   ```
   The **filepath** parameter and **testpath** parameter need to be set according to the path of the t2vec experimental data in the experimental environment.
   Cityname needs to be unified before and after, lon lat can refer to the same data set to fill in the same, cell size is the track cell width, the default is 100, file path and testpath are the paths of the two files obtained in preprocessing. Train_num, vocab_size are mainly for recording here, and these two parameters are not necessary (actually these two parameters should be filled in run_part1.sh run_part2.sh).

2. Then open run_part1.sh. The intermediate parameter train_num val_num is the number of trajectories used for training. This value needs to be determined according to the number of trajectories in train.h5. Train_num+val_num cannot exceed the number of trajectories in train.h5, starting from train.h5. The method of h5 to extract the number of trajectories is:

   ```python
   import h5py
   with h5py.File('train.h5','r') as f:
   print(f.attrs['traj_nums'])
   ```

   Other parameters are:

   ```bash
   region_name=region_geolife_len6.0 # region_name in hyper-parameters.json
   grid_size=180 # Unified with run_part1.sh
   train_num=3500
   val_num=500
   ```

   After setting the parameters, bash run_part1.sh

   Note that this bash file will output a vocab_size value, record it, my approach is to record it in hyper-parameters.json to save

3. Modify run_part2.sh to have the following parameters:

   ```bash
   vocab_size=10313
   layer=3 # t2vec model layer(default in porto: 3)
   cityname="geolife_pair6.0"
   grid_size=180 # need to be same as part 1
   region_name=region_geolife_len6.0
   gpu_id=0
   ```

   Cityname should be unified with hyper-parameters.json

   bash run_part2.sh will now train the embedding of the trajectory

4. Use traj_pair_exp.py to set the corresponding parameters and run the result

#### TULVAE

Code: [./traj_er/TULVAE](https://github.com/lzzppp/DERT/tree/master/traj_er/TULVAE)

Environment: [./traj_er/TULVAE/requirements.txt](https://github.com/lzzppp/DERT/tree/master/traj_er/TULVAE)

The code of this person is rather confusing. This code originally categorizes the trajectory. In order to deal with our problem, the method of running is to join the two trajectory cell sequences and run as a 2-classification problem.

The parameters of the main comparison algorithm have been basically set

```python
ap.add_argument("-dataset", type=str, default='singapore')
ap.add_argument("-length", type=float, default=2)
ap.add_argument("-gpu_id", type=str, default="0")
```

The above two parameters can be unified with other comparison algorithms, and the last parameter depends on the current idle gpu

If you want to run this comparison algorithm in other ways, you have to modify it

### Check-In-ER
The Check-In-ER task comparison algorithm is similar to the Traj-ER task comparison algorithm, but the task difficulty is different, so I won’t repeat it here.  
 
 
***
## contributing
Open source code and Dataset generation

***
## License
+ [MIT License](https://github.com/qyxxjd/License)


