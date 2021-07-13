import bifrost as bf
from copy import deepcopy
from datetime import datetime
from pprint import pprint
import time
from astropy.time import Time
import numpy as np
import os
#from models_network_architecture import models
from keras.models import load_model
from skimage.transform import resize
from models_network_architecture import models
import tensorflow as tf
from bifrost.ndarray import copy_array
### https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory ###
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#import sys, os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#from tensorflow.compat.v1.keras.backend import set_session
#config = tf.ConfigProto(device_count = {'GPU': 1}
#config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1})
#config.gpu_options.visible_device_list = "1"
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#config.gpu_options.allow_growth=True
#set_session(tf.compat.v1.Session(config=config))
from datetime import datetime
import matplotlib.pyplot as plt

#class get_data():
#    def 


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
#        self.weight_path = params['weight_path']
        self.data = params['data']
        self.model = params['model']
        self.total_beams = params['total_beams']
        self.time_shape = params['time_shape']
        self.prev_run = params['prev_run']
        self.n_iter = params['iter']

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
        y_pred_keras = np.zeros((2, self.batch_size, 2, self.time_shape // self.nsamps), dtype = np.float32)
        for j in range(self.time_shape // self.nsamps):
            y_pred_keras[0, :, :, j] = self.model.predict( [ self.normalise(self.data[0, :, :, (2048 * j) : (j + 1) * 2048, 0]).reshape(self.batch_size, 256, 2048, 1), self.normalise(self.data[0, :, :, (2048 * j) : (j + 1) * 2048, 1]).reshape(self.batch_size, 256, 2048, 1), self.normalise(self.data[0, :, :, (2048 * j) : (j + 1) * 2048, 2]).reshape(self.batch_size, 256, 2048, 1)])

        if self.n_iter != 0:
            for j in range(self.time_shape // self.nsamps):
                copy_array(self.data[0, :, :, 1024:], self.data[0, :, :, : (8192 - 1024)])
                copy_array(self.data[0, :, :, :1024], self.prev_run)
                y_pred_keras[1, :, :, j] = self.model.predict( [ self.normalise(self.data[0, :, :, (2048 * j) : (j + 1) * 2048, 0]).reshape(self.batch_size, 256, 2048, 1), self.normalise(self.data[0, :, :, (2048 * j) : (j + 1) * 2048, 1]).reshape(self.batch_size, 256, 2048, 1), self.normalise(self.data[0, :, :, (2048 * j) : (j + 1) * 2048, 2]).reshape(self.batch_size, 256, 2048, 1)])
        return y_pred_keras

class ABClassify(bf.pipeline.SinkBlock):
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
        self.batch_size = 8
#        self.model = load_model('/home/amandlik/ABC_direct_classifier/configa.hdf5') 

#        self.data_idx = 0
        #self._meta = {}
        self.freq_shape = itensor['shape'][itensor['labels'].index('freq')]
        self.time_shape = itensor['shape'][itensor['labels'].index('fine_time')]
        self.beam_shape = itensor['shape'][itensor['labels'].index('fine_beam')]
        self.data = bf.ndarray(shape = (1, self.batch_size, self.freq_shape, self.time_shape, self.beam_shape), space = 'system')
        self.prev_run = bf.ndarray(shape = (self.batch_size, self.freq_shape, 1024, self.beam_shape), dtype = np.float32, space = 'system')
        
        # Setup metadata
#        self._meta['source_name'] = ihdr['source_name']
#        self._meta['ra']          = ihdr['RA']
#        self._meta['dec']         = ihdr['DEC']
#        self._meta['labels']      = map(str, ihdr['_tensor']['labels'])
#        self._meta['units']       = map(str, ihdr['_tensor']['units'])
#        self._meta['scales']      = map(list, ihdr['_tensor']['scales'])


    def on_data(self, ispan):
#        num_tblock = 8
        data = ispan.data
        chunk_size = 2048
        #bf.reduce(data, self.freq_decimated, op= 'mean')
        #array = self.freq_decimated.split_axis()
        #array = array.transpose(()
        #bf.split_axis()
        #b_gpu = bf.views.split_axis(b_gpu, 'station', 2, label='pol')
        copy_array(self.data, data)


        params = {
            'nsamps' : chunk_size,
            'nchans' : self.freq_shape,
            'total_beams' : self.beam_shape,
            'time_shape' : self.time_shape,
            'fdecimate' : True,
            'batch_size' : self.batch_size,
            'num_beams' : 3,
            'tdecimate' : 1,
            'model_name' : 'concatenated_ABC',
            'model_dispersed' : True,
            'model' : self.model,
            'data': self.data,
            'prev_run': self.prev_run,
            'iter': self.n_iter}
  
        pred_obj = Direct_predict(params)
        #pred_obj.predict()
        predictions = pred_obj.predict()
        print("Direct classifier looking at block: ")
        print(self.n_iter)
        np.save('Direct_classifier_logs_' + str(self.n_iter), predictions)
        copy_array(self.prev_run, self.data[0, :, :, (8192 - 1024):])
        self.n_iter += 1
        
def adjbeam_classify(iring, model, outdir='./', prefix='abc-results', *args, **kwargs):
    return ABClassify(iring, model, outdir, prefix, *args, **kwargs)
