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
from datetime import datetime
import matplotlib.pyplot as plt
from glob import glob
import time


class FDMT_block(bf.pipeline.TransformBlock):
    def __init__(self, iring, *args, **kwargs):
        super(FDMT_block, self).__init__(iring, *args, **kwargs)
        self.kdm       = 4.148741601e3 # MHz**2 cm**3 s / pc
        self.dm_units  = 'pc cm^-3'
        self.timechunksize = 8192 
        self.n_iter = 0
        #self.timechunksize = 2048

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        refdm = 0
        dmstep = 1
        ohdr['name'] = 'FDMT_block'
        ohdr['_tensor']['dtype'] = 'f32'
        # The output can't be 8192x8192 - run out of memory! We will output a chunk.
        ohdr['_tensor']['shape'][-1]  = self.timechunksize
        ohdr['_tensor']['shape'][-2]  = self.timechunksize

        ohdr['_tensor']['shape'][1]  = 9 # for the number of beams across which to search for the pulse.
        #ohdr['_tensor']['shape'][-2]  = ohdr['_tensor']['shape'][-1]
        ohdr['_tensor']['labels'][-2] = 'dispersion'
        ohdr['_tensor']['scales'][-2] = (refdm, dmstep) # These need to be calculated properly
        ohdr['_tensor']['units'][-2]  = self.dm_units
        #self.reffreq = 856.25
        self.reffreq = 806.25
        self.freq_step = 100./1024
        self.f = bf.fdmt.Fdmt()
        self.predictions = np.load('Direct_classifier_logs_' + str(self.n_iter) + '.npy')
        self.furby_predictions = self.predictions[:, 1, :]
        self.beam_blocks = np.concatenate(np.argwhere( self.furby_predictions > 0.99 ))[::2] # Set the threshold values here
        ohdr['_tensor']['shape'][1]  = len(self.beam_blocks) * 3 # for the number of beams across which to search for the pulse.
        self.f.init(512, self.timechunksize, self.reffreq, self.freq_step)
        self.dm_values = bf.ndarray(shape = (self.timechunksize, 1), dtype = np.float32, space = 'cuda')

        self.bx_1 = bf.ndarray(shape = (self.timechunksize, self.timechunksize // 2), dtype = np.float32, space = 'cuda')
        self.bx_2 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 4), dtype = np.float32, space = 'cuda')
        self.bx_3 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 8), dtype = np.float32, space = 'cuda')
        self.bx_4 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 16), dtype = np.float32, space = 'cuda')
        self.bx_5 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 32), dtype = np.float32, space = 'cuda')
        self.bx_6 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 64), dtype = np.float32, space = 'cuda')
        self.bx_7 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 128), dtype = np.float32, space = 'cuda')
         
        self.box_c = {'1': self.bx_1, '2': self.bx_2, '3': self.bx_3, '4': self.bx_4, '5': self.bx_5, '6': self.bx_6, '7': self.bx_7}

        return ohdr
        

    def on_data(self, ispan, ospan):
        print('fdmt block')
        start_beam = 0 # get this from Direct classifier.
        in_nframe  = ispan.nframe
        odata = ospan.data
        out_nframe = in_nframe
        idata = ispan.data
        freq = np.arange(806.25, 856.25, 0.09765625)*1e6
        #ddata = bf.ndarray(shape = (8192, 8192), dtype = np.float32, space = 'cuda')
        #dm_values = bf.ndarray(shape = (8192, 1), dtype = np.float32, space = 'cuda')
        #time_values = bf.ndarray(shape = (1, 8192), dtype = np.float32, space = 'cuda')
        #print(ddata.shape)
        ### peak_values is array containing information about peak values in terms of (beam, boxcar, time_index, dm_index)

        for ob_index, beam_bl in enumerate(self.beam_blocks):
            # ob_index: output beam index.
            # beam_bl: the index of beam in the whole block with all beams.
            start = datetime.now()
            peak_values = np.zeros((3, 8, 3), dtype = np.float32)
            for beam_num in range(3):
                self.f.execute(idata[0, (3 * beam_bl) + beam_num, :, :self.timechunksize], odata[0, (3 * ob_index) + beam_num, :, :], negative_delays = True) 
                for box_car in range(8):
                    if box_car == 0:
                        tscrunched = odata[0, beam_num, :, :].copy(space = 'system')
                        peak_value = tscrunched.max()
                        peak_values[beam_num, box_car, 0] = peak_value
                        peak_values[beam_num, box_car, 1] = np.where(tscrunched == peak_value)[0][0]
                        peak_values[beam_num, box_car, 2] = np.where(tscrunched == peak_value)[1][0]
    
                    else:
                        bf.reduce(odata[0, beam_num, :, :], self.box_c[str(box_car)] , op = 'mean')
                        tscrunched = self.box_c[str(box_car)].copy(space = 'system')
                        peak_value = tscrunched.max()
                        norm_factor = np.sqrt(2**box_car)
                        peak_values[beam_num, box_car, 0] = peak_value * norm_factor
                        peak_values[beam_num, box_car, 1] = np.where(tscrunched == peak_value)[0][0]
                        time_index = np.where(tscrunched == peak_value)[1][0] * (2 ** box_car)
                        peak_values[beam_num, box_car, 2] = time_index
    
            final_beam = np.where(peak_values == peak_values[:,:,0].max())[0][0]
            final_box_car = np.where(peak_values == peak_values[:,:,0].max())[1][0]
            
            time = np.arange(0, (0.00032768 * self.timechunksize), 0.00032768)
            tint = time[1] - time[0]
            dm_ = np.arange(time.size)*1.0
            dm_ *= tint / 4.15e-3 / ((freq[0]/1e9)**-2 - (freq[-1]/1e9)**-2)
    
            # Way 1:
            dm_index = peak_values[final_beam, final_box_car, 1] 
            final_time = peak_values[final_beam, final_box_car, 2]
            final_dm = dm_[int(dm_index)]
            print(final_beam, final_box_car, final_dm, final_time)
            print(datetime.now() - start)
    
            # Way 2:
#            dm_index = np.median(peak_values[final_beam,:,1])
#            final_time = np.median(peak_values[final_beam,:,2])
#            final_dm = dm_[int(dm_index)]
#            print('Way 2')
#            print(final_beam, final_box_car, final_dm, final_time)
#            print(datetime.now() - start)
                   
#        peak_beam, peak_box_car = np.where(peak_values == peak_values.max())
#        print('peak_beam, peak_box_car')
#        print(peak_beam, peak_box_car)
                
#            dmt = odata[0, beam_num, :, :].copy(space = 'system')
        #    bf.reduce(odata[0,i,:,:], self.dm_values, op = 'max')
            #bf.reduce(odata[0,i,:,:], time_values, op = 'max')
#            dm_values = self.dm_values.copy(space = 'system')
#            time_values = self.time_values.copy(space = 'system')
#            peak_dm_loc.append(np.argmax(dm_values))
#            peak_dm_values.append(dm_values.max())

#        beam = np.argmax(peak_dm_values)
#        print(beam)
#        print(peak_dm_loc)
#        print(peak_dm_values)
#
#        peak_bxc_value = []
#        
#        for i in range(7):
#            if i == 0:
#                tscrunched = odata[0,beam,:,:].copy(space = 'system')
#                peak_value = tscrunched.max()
#                peak_bxc_value.append(peak_value)
#            else:
#                bf.reduce(odata[0,beam,:,:], self.box_c[str(i)] , op = 'mean')
#                tscrunched = self.box_c[str(i)].copy(space = 'system')
#                peak_value = tscrunched.max()
#                norm_factor = np.sqrt(2**i)
#                peak_bxc_value.append(peak_value * norm_factor)
#
#        print('Box car values' + str(peak_bxc_value))
#        box_car = np.argmax(peak_bxc_value)
#        print('Which boxcar ?' + str(box_car))
#
#
#        #self.f.execute(idata[0,0,:,:self.timechunksize], odata[0,0,:,:])
#        #f.execute(idata[0,0,:,256:512+256], odata[0,0,:,:])
#
#        time = np.arange(0, (0.00032768 * 8192), 0.00032768)
#        tint = time[1] - time[0]
#        dm_ = np.arange(time.size)*1.0
#        dm_ *= tint / 4.15e-3 / ((freq[0]/1e9)**-2 - (freq[-1]/1e9)**-2)
#        dm_index = np.median(peak_values[final_beam,:,1])
#        final_dm = dm_[dm_index]

#        #print(ddata.shape)
#        np.save('fdmt_blocksh_full' + str(self.block_count), odata[0,0,:,:])
#        np.save('fdmt_inputblock_full' + str(self.block_count), idata[0,0,:,:self.timechunksize])
#        np.save('fdmt_inputzz_rev' + str(self.block_count), idata[0,0,:,:self.timechunksize])
        self.n_iter += 1


def FDMT(iring, *args, **kwargs):
    return FDMT_block(iring, *args, **kwargs)
