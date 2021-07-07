import bifrost as bf
from copy import deepcopy
from datetime import datetime
from pprint import pprint
import time
from astropy.time import Time
import numpy as np
import os
from keras.models import load_model
from skimage.transform import resize
from models_network_architecture import models
import tensorflow as tf
### https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory ###
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
from datetime import datetime
import matplotlib.pyplot as plt

class Direct_predict():

    def __init__(self, params):
        
        self.nsamps = params['nsamps']
        self.nchans = params['nchans']
        self.fdecimate = params['fdecimate']
        self.batch_size = params['batch_size']
        self.num_beams = params['num_beams']
        self.tdecimate = params['tdecimate']
        self.model_name = params['model_name']
        self.model_dispersed = params['model_dispersed']
        self.array = params['array']
        self.model = params['model']
        self.total_beams = params['total_beams']
        self.time_shape = params['time_shape']

    def normalise(self, array):
         start = datetime.now()
         data = np.array(array, dtype=np.float32)
         data -= np.median(array)
         data /= np.std(data)
         return data

    def t_decimate(self, array):

        tdata = []
        for i in range(self.num_beams):
            data = array[:,:,i].T
            tdata_ = data.reshape(data.shape[0] , int(data.shape[1] // factor), factor).mean(axis = -1).T
            tdata.append(tdata_)
        print(np.shape(tdata))
        return np.dstack(tdata)


    def f_decimate(self, array, channels = 256):
        fdata = resize(array, (8, array.shape[0], channels, self.num_beams))
        return fdata


    def predict(self):
        start = datetime.now()
        y_pred_keras = np.empty((self.batch_size, 2, self.time_shape // self.nsamps), dtype = np.float32)
        #y_pred_keras = np.empty((self.total_beams // self.num_beams, self.time_shape // self.nsamps ))
        #data = self.array.reshape(1, self.nchans // 2, self.time_shape, self.batch_size, self.num_beams)
        #print('here 2')
        data = self.array.copy(space = 'system')
        #data = np.moveaxis(data, 3, 1)
        #print(data.shape)

        for j in range(self.time_shape // self.nsamps):
            y_pred_keras[:, :, j] = self.model.predict( [ self.normalise(data[0, :, :, (2048 * j) : (j + 1) * 2048, 0]).reshape(self.batch_size, 256, 2048, 1), 
                                                          self.normalise(data[0, :, :, (2048 * j) : (j + 1) * 2048, 1]).reshape(self.batch_size, 256, 2048, 1), 
                                                          self.normalise(data[0, :, :, (2048 * j) : (j + 1) * 2048, 2]).reshape(self.batch_size, 256, 2048, 1)] )
        return y_pred_keras

class ABC_dedispersed(bf.pipeline.SinkBlock):
    def __init__(self, iring, model, outdir='./', prefix='abc-results', *args, **kwargs):
        super(ABClassify, self).__init__(iring, *args, **kwargs)
        self.outdir = outdir
        self.seq_idx = 0
        self.n_iter = 0
        self.model = model

    def on_sequence(self, iseq):
        self.seq_idx += 1
        self.data_idx = 0
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        self.freq_shape = itensor['shape'][itensor['labels'].index('freq')]
        self.time_shape = itensor['shape'][itensor['labels'].index('fine_time')]
        self.beam_shape = itensor['shape'][itensor['labels'].index('fine_beam')]
        
    def on_data(self, ispan):
        data = ispan.data
        chunk_size = 2048
        batch_size = 8
        params = {
            'nsamps' : chunk_size,
            'nchans' : self.freq_shape,
            'total_beams' : self.beam_shape,
            'time_shape' : self.time_shape,
            'fdecimate' : True,
            'batch_size' : batch_size,
            'num_beams' : 3,
            'tdecimate' : 1,
            'model_name' : 'concatenated_ABC',
            'model_dispersed' : True,
            'model' : self.model,
            'array': data}
  
        pred_obj = Direct_predict(params)
        predictions = pred_obj.predict()
        print("Direct classifier looking at block: ")
        print(self.n_iter)
        np.save('Direct_classifier_logs_' + str(self.n_iter), predictions)

        self.n_iter += 1
        
def adjbeam_classify(iring, model, outdir='./', prefix='abc-results', *args, **kwargs):
    return ABC_dedispersed(iring, model, outdir, prefix, *args, **kwargs)
