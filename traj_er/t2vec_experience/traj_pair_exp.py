import numpy as np
import pandas as pd
import h5py
import sklearn
import os
import time
import keras 
from keras.models import Sequential 
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf
from classify_exp.classify_model import *
import os
import json
from trajvec_classify import *
os.environ["CUDA_VISIBLE_DEVICES"]="0" # model will be trained on which gpu
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
import argparse
import pickle
from keras import backend as K


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum (K.round (K.clip (y_true * y_pred, 0, 1)))
        possible_positives = K.sum (K.round (K.clip (y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon ())
        return recall
    
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum (K.round (K.clip (y_true * y_pred, 0, 1)))
        predicted_positives = K.sum (K.round (K.clip (y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon ())
        return precision
    
    precision = precision (y_true, y_pred)
    recall = recall (y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon ()))

def get_pairs(vec, idx_dict, pair_path):
    """给定轨迹vec和index映射和原始pair文件生成训练数据和label"""
    df = pd.read_csv(pair_path)
    labels = df['label'].values
    origin_pair_idxs = df[['ltable_id', 'rtable_id']].values
    data = [np.concatenate((vec[idx_dict[l]], vec[idx_dict[r]]), axis=0) for l,r in origin_pair_idxs]
    return np.array(data), labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trajectory pair match experiment")

    parser.add_argument("-grid_size", type=int, default=180, help="Grid size")
    # parser.add_argument("-cityname", type=str, help="Dataset name")
    parser.add_argument("-emb_size", type=int, default=256, help="Embedding Size")
    parser.add_argument("-num_layers", type=int, default=3, help="Embeding on which layer")
    # parser.add_argument("-cityname", type=str, help="city name")
    parser.add_argument("-length", type=int)
    # parser.add_argument("-save_outcome", type=bool, default=False, help="if save the predict outcome for analysis")

    parser.add_argument("-dataset", type=str, default="porto")

    args = parser.parse_args()

    # base_folder = os.path.join('/home/lizepeng/traj_data/t2vec/singapore')
    # pair_folder = os.path.join('/home/lizepeng/traj_data/mcan/singapore')
    base_folder = os.path.join('/home/lizepeng/new_data_24h/t2vec/geolife/24.0h')
    pair_folder = os.path.join('/home/lizepeng/new_data_24h/geolife/24.0h')
    # cityname = hyper_param[args.region_name]["cityname"]
    cityname = "%s_len%d" % (args.dataset, args.length)
    vec_folder = 'experiment/traj_emb'
    # train_vec = get_emb(os.path.join(vec_folder, "trj_{}_train{}.h5".format(cityname, args.grid_size)), args.num_layers)
    # test_vec = get_emb(os.path.join(vec_folder, "trj_{}_test{}.h5".format(cityname, args.grid_size)), args.num_layers)
    
    train_vec = get_emb(os.path.join(vec_folder, "trj_geolife_pair24.0_train180.h5"), args.num_layers)
    test_vec = get_emb(os.path.join(vec_folder, "trj_geolife_pair24.0_test180.h5"), args.num_layers)

    # Generate real train test pairs
    with open(os.path.join(base_folder, 'train_dict.pkl'), 'rb') as f:
        train_dict = pickle.load(f)
    with open(os.path.join(base_folder, 'test_dict.pkl'), 'rb') as f:
        test_dict = pickle.load(f)

    X_train, y_train = get_pairs(train_vec, train_dict, os.path.join(pair_folder, 'train.csv'))
    X_test, y_test = get_pairs(test_vec, test_dict, os.path.join(pair_folder, 'test.csv'))


    mid_size = 512
    input_size = 512
    num_classes = 2
    nn_model = Sequential([
                Dense(mid_size, activation='relu', input_shape=(input_size,)),
                Dense(1, activation='sigmoid')
    ])

    adam = keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    nn_model.compile(
        optimizer=adam,
        loss='binary_crossentropy',
        # metrics=['accuracy'],
        metrics=['accuracy', f1]
    )

    start_time = time.time()
    his = nn_model.fit(X_train, y_train, epochs=20, batch_size=64,
             validation_data=None, verbose=2)
    end_time = time.time()
    run_time = end_time - start_time
    print('run time is : ', run_time)
    evaluate_result = nn_model.evaluate(X_test, y_test, batch_size=32)
    print(evaluate_result)
    start_time = time.time()
    print('----------------')
    print(X_test.shape)
    print('----------------')
    prediction = nn_model.predict(X_test, batch_size=32)
    predicts = []
    pos_num, neg_num = 0, 0
    for line in prediction:
        if line[0] > 0.5:
            pos_num += 1
            predicts.append(1)
        elif line[0] < 0.5:
            neg_num += 1
            predicts.append(0)
    end_time = time.time()
    test_time = end_time - start_time
    print('test time is : ', test_time)
    TP, FP, TN, FN = 0, 0, 0, 0
    num = 0
    for target, predict in zip (y_test, predicts):
        if int(target) == 1 and predict == 1:
            TP += 1
        elif int(target) == 0 and predict == 1:
            FP += 1
        elif int(target) == 0 and predict == 0:
            TN += 1
        elif int(target) == 1 and predict == 0:
            FN += 1
    PP = TP / (TP + FP)
    NP = TN / (FN + TN)
    PT = TP / (TP + FN)
    NT = TN / (FP + TN)
    print ("POS P value is : ", TP / (TP + FP))
    print ("NEG P value is : ", TN / (FN + TN))
    print ("POS T value is : ", TP / (TP + FN))
    print ("NEG T value is : ", TN / (FP + TN))
    print ("POS F1 value is : ", 2 * PP * PT / (PP + PT))
    print ("NEG F1 value is : ", 2 * NP * NT / (NP + NT))
    # file = open("result.pkl", "wb")
    # pickle.dump(prediction, file)
    # file_test = open("result_test.pkl", "wb")
    # pickle.dump(y_test, file_test)
    print("Best F1-score : %.2f" % (2 * PP * PT / (PP + PT) * 100))
