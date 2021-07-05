from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, concatenate, Conv3D, Activation, MaxPooling3D, BatchNormalization, Dropout
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.optimizers import SGD, Adam
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import pandas as pd
import argparse
import glob
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
import pickle
import sklearn.metrics as metrics
import gc
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from keras.callbacks import ModelCheckpoint
from keras.utils.layer_utils import count_params

def models(width, height, depth, model_name, k_size, p_size, filter1, filter2, filter3, s_size = 1, regulariser_=None, dropout_=None):

    if model_name == 'Traditional_ABC':

        def create_cnn(width, height, depth, filters=(filter1, filter2, filter3), regress=False):
            # initialize the input shape and channel dimension
            inputShape = (height, width, depth)
            chanDim = -1
    
        # define the model input
            inputs = Input(shape=inputShape)
    
        # loop over the number of filters
            for (i, f) in enumerate(filters):
            # if this is the first CONV layer then set the input
            # appropriately
                if i == 0:
                    x = inputs
    
            # CONV => RELU => BN => POOL
                x = Conv2D(f, (k_size, k_size), strides=(s_size, s_size), padding="same")(x)
                x = Activation("relu")(x)
                x = BatchNormalization(axis=chanDim)(x)
                x = MaxPooling2D(pool_size=(p_size, p_size), padding="same")(x)
    
        # flatten the volume, then FC => RELU => BN => DROPOUT
            x = Flatten()(x)
            x = Dense(16)(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)
    
        # apply another FC layer, this one to match the number of nodes coming out of the MLP
            x = Dense(4)(x)
            x = Activation("relu")(x)
    
        # construct the CNN
            model = Model(inputs, x)
    
        # return the CNN
            return model
    
        freq_time_cnn = create_cnn(width, height, depth, regress=False)
        num_labels=2
        x = Dense(4, activation="relu")(freq_time_cnn.output)
        if dropout_ != None:
            rate = dropout_
            x = tf.keras.layers.Dropout(rate)(x)
        x = Dense(num_labels, activation="softmax")(x)
        model = Model(inputs=[freq_time_cnn.input], outputs=x)
        if regulariser_ != None:
            penalty = regulariser_
            regularizer = tf.keras.regularizers.l2(penalty)
            for layer in model.layers:
                for attr in ['kernel_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)
        return model

    
    if model_name == 'concatenated_ABC':

        def create_cnn(width, height, depth, filters=(filter1, filter2, filter3), regress=False):
            # initialize the input shape and channel dimension
            inputShape = (height, width, depth)
            chanDim = -1
        
            # define the model input
            inputs = Input(shape=inputShape)
        
            # loop over the number of filters
            for (i, f) in enumerate(filters):
            # if this is the first CONV layer then set the input
            # appropriately
                if i == 0:
                    x = inputs
        
                # CONV => RELU => BN => POOL
                x = Conv2D(f, (k_size, k_size), strides=(s_size, s_size), padding="same")(x)
                x = Activation("relu")(x)
                x = BatchNormalization(axis=chanDim)(x)
                x = MaxPooling2D(pool_size=(p_size, p_size), padding="same")(x)
        
            # flatten the volume, then FC => RELU => BN => DROPOUT
            x = Flatten()(x)
            x = Dense(16)(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)
 #           x = Dropout(0.5)(x)
        
            # apply another FC layer, this one to match the number of nodes coming out of the MLP
            x = Dense(4)(x)
            x = Activation("relu")(x)
        
            # construct the CNN
            model = Model(inputs, x)
            return model

        if depth == 5:
            num_labels = 2
        
            model_beam_1 = create_cnn(width, height, 1, regress=False)
            model_beam_2 = create_cnn(width, height, 1, regress=False)
            model_beam_3 = create_cnn(width, height, 1, regress=False)
            model_beam_4 = create_cnn(width, height, 1, regress=False)
            model_beam_5 = create_cnn(width, height, 1, regress=False)
            combinedInput = concatenate([model_beam_1.output,model_beam_2.output, model_beam_3.output, model_beam_4.output, model_beam_5.output])
            x = Dense(4, activation="relu")(combinedInput)
            if dropout_ != None:
                rate = dropout_
                x = tf.keras.layers.Dropout(rate)(x)
            x = Dense(num_labels, activation="softmax")(x)
            model = Model(inputs=[model_beam_1.input, model_beam_2.input, model_beam_3.input, model_beam_4.input, model_beam_5.input], outputs=x)
            if regulariser_ != None:
                penalty = regulariser_
                regularizer = tf.keras.regularizers.l2(penalty)
                for layer in model.layers:
                    for attr in ['kernel_regularizer']:
                        if hasattr(layer, attr):
                            setattr(layer, attr, regularizer)
        if depth == 4:
            num_labels = 2
            model_beam_1 = create_cnn(width, height, 1, regress=False)
            model_beam_2 = create_cnn(width, height, 1, regress=False)
            model_beam_3 = create_cnn(width, height, 1, regress=False)
            model_beam_4 = create_cnn(width, height, 1, regress=False)
            combinedInput = concatenate([model_beam_1.output,model_beam_2.output, model_beam_3.output, model_beam_4.output])
            x = Dense(4, activation="relu")(combinedInput)
            if dropout_ != None:
                rate = dropout_
                x = tf.keras.layers.Dropout(rate)(x)
            x = Dense(num_labels, activation="softmax")(x)
            model = Model(inputs=[model_beam_1.input, model_beam_2.input, model_beam_3.input, model_beam_4.input], outputs=x)
            if regulariser_ != None:
                penalty = regulariser_
                regularizer = tf.keras.regularizers.l2(penalty)
                for layer in model.layers:
                    for attr in ['kernel_regularizer']:
                        if hasattr(layer, attr):
                            setattr(layer, attr, regularizer)

        if depth == 3:
            num_labels = 2
            model_beam_1 = create_cnn(width, height, 1, regress=False)
            model_beam_2 = create_cnn(width, height, 1, regress=False)
            model_beam_3 = create_cnn(width, height, 1, regress=False)
            combinedInput = concatenate([model_beam_1.output,model_beam_2.output, model_beam_3.output])
            x = Dense(4, activation="relu")(combinedInput)
            if dropout_ != None:
                rate = dropout_
                x = tf.keras.layers.Dropout(rate)(x)
            x = Dense(num_labels, activation="softmax")(x)
            model = Model(inputs=[model_beam_1.input, model_beam_2.input, model_beam_3.input], outputs=x)
            if regulariser_ != None:
                penalty = regulariser_
                regularizer = tf.keras.regularizers.l2(penalty)
                for layer in model.layers:
                    for attr in ['kernel_regularizer']:
                        if hasattr(layer, attr):
                            setattr(layer, attr, regularizer)

        return model

