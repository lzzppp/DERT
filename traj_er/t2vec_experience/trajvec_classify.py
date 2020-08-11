import numpy as np
import pandas as pd
import h5py
import sklearn
import os
import keras 
from keras.models import Sequential 
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf
from classify_exp.classify_model import *
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"]="0" # model will be trained on GPU 1
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
import argparse

def get_label(dataset_args):
    def extract_label(path):
        label = []
        with h5py.File(path, 'r') as f:
            num = f.attrs['traj_nums']
            print(num)
            for i in range(num):
                label.append(int(np.array(f['taxi_ids/%d'%i])))
            return np.array(label)
    if os.path.exists(dataset_args["trainlabel"]):
        return np.load(dataset_args["trainlabel"]), np.load(dataset_args["testlabel"])
    else:
        train_label = extract_label(dataset_args["filepath"])
        test_label = extract_label(dataset_args["testpath"])
        np.save(dataset_args["trainlabel"], train_label)
        np.save(dataset_args["testlabel"], test_label)
        return train_label, test_label

def get_emb(path, layer):
    with h5py.File(path, 'r') as f:
        vec = np.array(f['layer%d' % layer])
    return vec

# def get_model(input_size, mid_size, classify_num):
#     nn_model = Sequential([
#             Dense(mid_size, activation='relu', input_shape=(input_size,)),
#             Dense(mid_size, activation='relu'),
#             Dense(classify_num, activation='softmax')
#     ])
#     return nn_model

def save_predict_outcome(path, true_label, pred_outcome):
    assert len(true_label) == len(pred_outcome), 'label and outcome should be the same size'
    with open(path, 'w') as f:
        out_string = ""
        for i in range(len(true_label)):
            out_string += str(true_label[i]) + ' ' + str(pred_outcome[i]) + '\n'
            if i % 1000 == 0 and i != 0:
                f.write(out_string)
                out_string = ""

def generate_file(city, embeddings, labels):
    assert len(embeddings) == len(labels)
    out = ""
    f = open(os.path.join('analysis_file', city+'.txt'), 'w')
    for idx, label in enumerate(labels):
        out += str(label) + " "
        emb = embeddings[idx]
        out += ",".join(list(map(str, emb)))
        out += '\n'
        if idx % 1000 == 0:
            f.write(out)
            out = ""

    f.close()





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="trajvec_classify.py")

    parser.add_argument("-grid_size", type=int, default=180, help="Grid size")
    # parser.add_argument("-cityname", type=str, help="Dataset name")
    parser.add_argument("-emb_size", type=int, default=256, help="Embedding Size")
    parser.add_argument("-num_layers", type=int, default=3, help="Embeding on which layer")
    parser.add_argument("-region_name", type=str, help="region name")
    parser.add_argument("-additional_feature", type=bool, default=False, help="test on additional feature(trip time, speed, distance...)")
    parser.add_argument("-save_outcome", type=bool, default=False, help="if save the predict outcome for analysis")

    args = parser.parse_args()
    with open('hyper-parameters.json', 'r') as f:
        hyper_param = json.loads(f.read())
    cityname = hyper_param[args.region_name]["cityname"]
    y_train, y_test = get_label(hyper_param[args.region_name])
    # y_train = get_label(hyper_param[args.region_name]["filepath"])
    # y_test = get_label(hyper_param[args.region_name]["testpath"])
    driver_num = y_train[-1] + 1
    vec_folder = 'experiment/traj_emb'
    X_train = get_emb(os.path.join(vec_folder, "trj_{}_train{}.h5".format(cityname, args.grid_size)), args.num_layers)
    X_test = get_emb(os.path.join(vec_folder, "trj_{}_test{}.h5".format(cityname, args.grid_size)), args.num_layers)
    generate_file(cityname, X_train, y_train)
    exit()
    # print(X_train.shape)
    # print(X_train[0])
    # print(X_test.shape)
    # exit()
    if not os.path.exists('saved_classify_model'):
        os.mkdir('saved_classify_model')

    if not os.path.exists('saved_pred_outcome'):
        os.mkdir('saved_pred_outcome')
    saved_classify_model = 'saved_classify_model/' + cityname + str(args.additional_feature) + 'best.h5'

    lr_set = [0.00005]
    # lr_set = []
    best_acc = 0.0
    best_lr = 0

    if args.additional_feature:
        # if use [19:], it means that only test speed distance and time feature
        train_add_feature = np.load(hyper_param[args.region_name]["trainfeature"])[:, 19:]
        add_feature_size = train_add_feature.shape[1]
        X_train = [X_train, train_add_feature]
        test_add_feature = np.load(hyper_param[args.region_name]["testfeature"])[:, 19:]
        X_test = [X_test, test_add_feature]

    for lr in lr_set:
        print('Start training on learning rate %f' % lr)
        # nn_model = Sequential([
        #     Dense(512, activation='relu', input_shape=(args.emb_size,)),
        #     Dense(512, activation='relu'),
        #     Dense(driver_num, activation='softmax')
        # ])
        if not args.additional_feature:
            nn_model = get_model(args.emb_size, 512, driver_num)
        else:
            nn_model = get_added_feature_model2(args.emb_size, add_feature_size, 512, driver_num)
        adam = keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
        nn_model.compile(
            optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        mc = keras.callbacks.ModelCheckpoint(saved_classify_model, monitor='val_acc',save_best_only=True, save_weights_only=True)  
        history = nn_model.fit(X_train, to_categorical(y_train), epochs=50, batch_size=64, validation_data=(X_test, to_categorical(y_test)), verbose=0, callbacks=[mc])

        lr_acc = max(history.history['val_acc'])
        if lr_acc > best_acc:
            best_acc = lr_acc
            best_lr = lr
    
    

    if args.save_outcome:
        if not args.additional_feature:
            best_model = get_model(args.emb_size, 512, driver_num)
        else:
            best_model = get_added_feature_model2(args.emb_size, add_feature_size, 512, driver_num)
        best_model.load_weights(saved_classify_model)

        out_come = np.argmax(best_model.predict(X_test, batch_size=64), axis=1)
        print(sum(out_come == y_test) / len(out_come))

        save_predict_outcome('saved_pred_outcome/' + cityname + str(args.additional_feature) + '.txt', y_test, out_come)
        # print(sum(y_test == out_come) / len(y_test))  # test my outcome is true

    print('Best test accuracy %f appear on learning rate %f' % (best_acc, best_lr))