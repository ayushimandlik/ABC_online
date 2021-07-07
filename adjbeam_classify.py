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
### https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory ###
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#import sys, os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#from tensorflow.compat.v1.keras.backend import set_session
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.visible_device_list = "0"
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
        self.array = params['array']
        self.model = params['model']
        self.total_beams = params['total_beams']
        self.time_shape = params['time_shape']

    def normalise(self, array):
        
#        if self.model_name == 'Traditional_ABC' or self.model_name == '3DCNN':
#            full_arr = []
#            for i in range(self.num_beams):
#                ###### Check if dtype would make a difference #######
#                data = np.array(array[:,:,i],dtype=np.float32)
#                data -= np.median(data)
#                data /= np.std(data)
#                full_arr.append(data)
#            fin = np.dstack(full_arr)
#            return fin
#        if self.model_name == ['concatenated_ABC']:
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

    def get_data(self, array):
#        tot_samps = array.shape[0]
#        samps_taken = array.shape[0] - (array.shape[0] % self.nsamps)
#        beams_taken = array.shape[-1] - (array.shape[-1] % self.num_beams)
#        print('Number of samples taken: ' + str(samps_taken))
#        print('Number of beams taken: ' + str(beams_taken))
#
#  
#        data = array[:samps_taken, : ,:beams_taken].reshape(8, 2048, 512, beams_taken //  self.num_beams, self.num_beams).swapaxes(3, 4)   

#        data = array.reshape(self.batch_size, self.num_samps, array.shape[1], array.shape[2]).reshape(self.batch_size, self.num_samps, array.shape[1], array.shape[2], )
#        data = 
#        tot_samps = array.shape[0]
#        batch_i = tot_samps // self.nsamps
#        batch_j = array.shape[-1] // self.num_beams
#        data = np.empty((batch_i * batch_j, self.nsamps, array.shape[1], self.num_beams))
#        self.batch_size = batch_i * batch_j
#        x = 0
#        for i in range(batch_i):
#            for j in range(batch_j):
#                print(array[i:i+self.nsamps, :, j:self.num_beams+j].shape)
#                data[x, ] = array[i:i+self.nsamps, :, j:self.num_beams+j]
#                print(x)
#                x = x + 1
#                print(data.shape)
        print(array.shape)
        print('get_data')
        data = array[:,:,:256,:3]
        print(data.shape)
        data = data.reshape(8, 2048, 256, 3)
#        print(data[:,0,0,0])
        print(data.shape)
        return data
#        data = self.normalise(data)
   
#        if self.fdecimate:
#            data = self.f_decimate(data)
#        if self.tdecimate != 1:
#            data = self.t_decimate(data)
#        if model_dispersed:
#         return data
#        else:
#            ft_data = self.dedisperse(data)
#            dm_data = self.dmtime(data)

    def predict(self):
#        array = self.get_data(self.array)
#        print('Got data')
        #print(data.shape)
#        model = models(self.nsamps // self.tdecimate, self.nchans, self.num_beams, self.model_name, 3, 2, 64, 32, 16) 
#        model.summary()
        ########
#        model = load_model(self.weight_path)
#        model.summary()
        ########
        #data = bf.ndarray(array, space='system')
#        for i in range(5):
#            start = datetime.now()
#            data = self.array.copy(space='system')
#            print('To put in system memory')
#            print(datetime.now() - start)
        #data = self.array.copy(space='system')
        #tf_data = tf.constant(self.array, dtype=tf.float32, name='data')

#        for i in range(5):
#            start = datetime.now()
#            data1 = (data[:,::2,:,:] +  data[:,1::2, :, :]) /2
#            print('For averaging')
#            print(datetime.now() - start)

        #print(tf_data)
        #for i in range(5):
        start = datetime.now()
        #bf.reduce(self.array, self.freq_decimated, op= 'mean')
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
        #print('For prediction')
        #print(datetime.now() - start)
        #print(y_pred_keras)
        #print(y_pred_keras.shape)
        return y_pred_keras
#        print(data.shape)
        #data = data.reshape(data.shape[0] , int(data.shape[1] // 2), data.shape[2],  data.shape[3], 2).mean(axis = 1)
        #d = np.random.randn(256, 2048, 3)
        #x = bf.ndarray(d, space = 'cuda')
        #data = b
        #print('predict')
        #print(data.shape)

        #print(x[:,:,0].shape)
#        model = self.model
#        y_pred_keras = model.predict([data[:,:,:,0].reshape(8, 256, 2048, 1), data[:,:,:,1].reshape(8, 256, 2048, 1), data[:,:,:,2].reshape(8, 256, 2048, 1)])
#        duration = datetime.now() - start
#        print(duration)
    #    num_cands = 8 * (data1.shape[-1] // 3)
    #    print(num_cands)

    #    predictions_rfi = [item[0] for item in np.array(y_pred_keras).reshape(num_cands, -1)]
    #    print(y_pred_keras)
    #    print(predictions_rfi)

    #    positives = np.where(np.array(predictions_rfi) < 0.9999)
    #    [i for i in range(len(k)) if k[i] > 2]    
    #    beam_numbers = [i for i in positives % 3]
    #    candidate_chunk = (positives % 8) * 2048

    #    tstart = candidate_chunk - (256 * 128)
    #    tstop = candidate_chunk + (256 * 128) + 2048
    #    print(positives)
        return y_pred_keras 

#class ABClassify(bf.pipeline.TransformBlock):
class ABClassify(bf.pipeline.SinkBlock):
    def __init__(self, iring, model, outdir='./', prefix='abc-results', *args, **kwargs):
        super(ABClassify, self).__init__(iring, *args, **kwargs)
        self.outdir = outdir
        self.seq_idx = 0
        self.n_iter = 0
        #TODO (what is data.idx???)
#        self.data_idx = 0
        #self._meta = {}
        self.model = model

    def on_sequence(self, iseq):
        self.seq_idx += 1
        self.data_idx = 0
        ihdr = iseq.header
        itensor = ihdr['_tensor']

#        self.data_idx = 0
        #self._meta = {}
        self.freq_shape = itensor['shape'][itensor['labels'].index('freq')]
        self.time_shape = itensor['shape'][itensor['labels'].index('fine_time')]
        self.beam_shape = itensor['shape'][itensor['labels'].index('fine_beam')]
        
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
        batch_size = 8
        #bf.reduce(data, self.freq_decimated, op= 'mean')
        #array = self.freq_decimated.split_axis()
        #array = array.transpose(()
        #bf.split_axis()
        #b_gpu = bf.views.split_axis(b_gpu, 'station', 2, label='pol')
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
    return ABClassify(iring, model, outdir, prefix, *args, **kwargs)
