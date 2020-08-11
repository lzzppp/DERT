import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.utils import to_categorical
import keras


def get_model(input_size, mid_size, classify_num):
    nn_model = Sequential([
            Dense(mid_size, activation='relu', input_shape=(input_size,)),
            Dense(mid_size, activation='relu'),
            Dense(classify_num, activation='softmax')
    ])
    return nn_model

def get_added_feature_model(input_size, add_feature_size, mid_size, classify_num):
    traj_vec_input = Input(shape=(input_size,), name="traj_vec_input")
    added_feature_input = Input(shape=(add_feature_size,), name="added_fature_input")
    x = keras.layers.concatenate([traj_vec_input, added_feature_input])
    x = Dense(mid_size, activation='relu')(x)
    x = Dense(mid_size, activation='relu')(x)
    output = Dense(classify_num, activation='softmax', name="output")(x)
    model = Model(inputs=[traj_vec_input, added_feature_input], outputs=output)
    return model

def get_added_feature_model2(input_size, add_feature_size, mid_size, classify_num):
    traj_vec_input = Input(shape=(input_size,), name="traj_vec_input")
    added_feature_input = Input(shape=(add_feature_size,), name="added_fature_input")
    added_feature_middle = Dense(30, activation='relu')(added_feature_input)
    x = keras.layers.concatenate([traj_vec_input, added_feature_middle])
    x = Dense(mid_size, activation='relu')(x)
    x = Dense(mid_size, activation='relu')(x)
    output = Dense(classify_num, activation='softmax', name="output")(x)
    model = Model(inputs=[traj_vec_input, added_feature_input], outputs=output)
    return model