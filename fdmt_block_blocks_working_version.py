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
from bifrost.ndarray import copy_array

class FDMT_block(bf.pipeline.TransformBlock):
    def __init__(self, iring, *args, **kwargs):
        super(FDMT_block, self).__init__(iring, *args, **kwargs)
        self.kdm       = 4.148741601e3 # MHz**2 cm**3 s / pc
        self.dm_units  = 'pc cm^-3'
        self.timechunksize = 8192 
        self.n_iter = 0

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        refdm = 0
        dmstep = 1
        ohdr['name'] = 'FDMT_block'
        ohdr['_tensor']['dtype'] = 'f32'
        ohdr['_tensor']['shape'][-1]  = self.timechunksize
        ohdr['_tensor']['shape'][-2]  = self.timechunksize
        ohdr['_tensor']['labels'][-2] = 'dispersion'
        ohdr['_tensor']['scales'][-2] = (refdm, dmstep) # These need to be calculated properly
        ohdr['_tensor']['units'][-2]  = self.dm_units
        self.reffreq = 806.25
        self.freq_step = 100./1024
        self.f = bf.fdmt.Fdmt()
        ohdr['_tensor']['shape'][1]  = 9 # for the number of beams across which to search for the pulse.
        self.f.init(512, self.timechunksize, self.reffreq, self.freq_step)

        self.dm_values = bf.ndarray(shape = (self.timechunksize, 1), dtype = np.float32, space = 'cuda')
        self.block_N_1 = bf.ndarray(shape = (1, 24, 512, 8192), dtype = np.float32, space = 'system') 
        self.block_N_2 = bf.ndarray(shape = (1, 24, 512, 8192), dtype = np.float32, space = 'system') 
#        self.peak_value = bf.ndarray(shape = (1, 1), dtype = np.float32, space = 'system') 


#        self.block_N_1 = bf.ndarray(shape = iseq.header['_tensor']['shape'], dtype = np.float32, space = 'cuda')
#        self.block_N_2 = bf.ndarray(shape = iseq.header['_tensor']['shape'], dtype = np.float32, space = 'cuda')

        self.bx_0 = bf.ndarray(shape = (self.timechunksize, self.timechunksize), dtype = np.float32, space = 'cuda')
        self.bx_1 = bf.ndarray(shape = (self.timechunksize, self.timechunksize // 2), dtype = np.float32, space = 'cuda')
        self.bx_2 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 4), dtype = np.float32, space = 'cuda')
        self.bx_3 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 8), dtype = np.float32, space = 'cuda')
        self.bx_4 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 16), dtype = np.float32, space = 'cuda')
        self.bx_5 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 32), dtype = np.float32, space = 'cuda')
        self.bx_6 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 64), dtype = np.float32, space = 'cuda')
        self.bx_7 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 128), dtype = np.float32, space = 'cuda')
        self.box_c = {'0': self.bx_0, '1': self.bx_1, '2': self.bx_2, '3': self.bx_3, '4': self.bx_4, '5': self.bx_5, '6': self.bx_6, '7': self.bx_7}

        self.bx_0_ts = bf.ndarray(shape = (self.timechunksize, self.timechunksize), dtype = np.float32, space = 'system')
        self.bx_1_ts = bf.ndarray(shape = (self.timechunksize, self.timechunksize // 2), dtype = np.float32, space = 'system')
        self.bx_2_ts = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 4), dtype = np.float32, space = 'system')
        self.bx_3_ts = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 8), dtype = np.float32, space = 'system')
        self.bx_4_ts = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 16), dtype = np.float32, space = 'system')
        self.bx_5_ts = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 32), dtype = np.float32, space = 'system')
        self.bx_6_ts = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 64), dtype = np.float32, space = 'system')
        self.bx_7_ts = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 128), dtype = np.float32, space = 'system')
        self.box_c_ts = {'0': self.bx_0_ts, '1': self.bx_1_ts, '2': self.bx_2_ts, '3': self.bx_3_ts, '4': self.bx_4_ts, '5': self.bx_5_ts, '6': self.bx_6_ts, '7': self.bx_7_ts}
        return ohdr
        
    def on_data(self, ispan, ospan):
        
#        start_beam = 0 # get this from Direct classifier.
#        if self.n_iter == 0:
#            print('on_data of fdmt niter is 0')
#            copy_array(self.block_N_2, ispan.data)
#        elif self.n_iter == 1:
#            print('on_data of fdmt niter is 1')
#            copy_array(self.block_N_1, ispan.data)
#            self.block_N_1 = bf.copy_array(ispan.data, space = 'cuda')
#        if self.n_iter >= 2:
            self.predictions = np.load('Direct_classifier_logs_' + str(self.n_iter - 1) + '.npy')
            self.furby_predictions = self.predictions[:, 1, :]
            self.beam_blocks = [] 
            for beam in sorted(np.ravel(self.furby_predictions), reverse = True)[:3]:
                self.beam_blocks.append(np.concatenate(np.argwhere( self.furby_predictions == beam ))[::2]) # Set the threshold values here

            in_nframe  = ispan.nframe
            print(ospan.data.shape)
            odata = ospan.data
            out_nframe = in_nframe
            idata = bf.ndarray.copy(self.block_N_1, space = 'cuda')
            freq = np.arange(806.25, 856.25, 0.09765625)*1e6
            print(self.beam_blocks)

            ### peak_values is array containing information about peak values in terms of (beam, boxcar, time_index, dm_index)
            for ob_index, beam_bl in enumerate(self.beam_blocks):
                # ob_index: output beam index.
                # beam_bl: the index of beam in the whole block with all beams.
                start = datetime.now()
                peak_values = np.zeros((3, 8, 3), dtype = np.float32)

                for beam_num in range(3):
                    print(beam_bl, beam_num, ob_index)
                    self.f.execute(idata[0, (3 * beam_bl[0]) + beam_num, :, :self.timechunksize], odata[0, (3 * ob_index) + beam_num, :, :], negative_delays = True) 
                    print(odata.shape)
                    for box_car in range(8):
                        if box_car == 0:
                            print('ssup')
                            print((3 * ob_index) + beam_num)
                            print(self.box_c_ts[str(box_car)].shape)
                            print(odata[0, (3 * ob_index) + beam_num, :, :].shape)
#                            print(odata[0, (3 * ob_index) + beam_num, :, :].max())
                            copy_array(self.box_c_ts[str(box_car)], odata[0, (3 * ob_index) + beam_num, :, :])
                    #        print(odata[0, (3 * ob_index) + beam_num, :, :].shape)
                    #        print(self.box_c_ts[str(box_car)].shape)
                    #        print(self.box_c_ts[str(box_car)][:,:,(3 * ob_index) + beam_num].shape)
#                            copy_array(self.box_c_ts[str(box_car)][:,:,(3 * ob_index) + beam_num], odata[0, (3 * ob_index) + beam_num, :, :])
                            peak_value = self.box_c_ts[str(box_car)].max()
                            print(peak_value)
#                            [:,:,(3 * ob_index) + beam_num].max()
#                            print(peak_value)
#                            tscrunched = odata[0, (3 * ob_index) + beam_num, :, :]
             #               tscrunched = odata[0, beam_num, :, :].copy(space = 'system')
                            #print(odata[0, (3 * ob_index) + beam_num, :, :].max)
                            #copy_array(self.tscrunched, odata[0, (3 * ob_index) + beam_num, :, :])
                            #peak_value = self.box_c_ts[str(box_car)].max()
                            peak_values[beam_num, box_car, 0] = peak_value
                            peak_values[beam_num, box_car, 1] = np.where(self.bx_0_ts == peak_value)[0][0]
                            peak_values[beam_num, box_car, 2] = np.where(self.bx_0_ts == peak_value)[1][0]
                            print(peak_values[beam_num, box_car, 0], peak_values[beam_num, box_car, 1], peak_values[beam_num, box_car, 2])
        
                        else:
                            bf.reduce(odata[0, (3 * ob_index) + beam_num, :, :], self.box_c[str(box_car)] , op = 'mean')
                            print('Here')
                            print(self.box_c_ts[str(box_car)].shape)
                            print(self.box_c[str(box_car)].shape)
                            #print(self.box_c[str(box_car)].max())
#                            print(self.box_c[str(box_car)].max())
                            copy_array(self.box_c_ts[str(box_car)], self.box_c[str(box_car)])
                            peak_value = self.box_c_ts[str(box_car)].max()
                            print(peak_value)
                            norm_factor = np.sqrt(2**box_car)
                            peak_values[beam_num, box_car, 0] = peak_value * norm_factor
                            peak_values[beam_num, box_car, 1] = np.where(self.box_c_ts[str(box_car)] == peak_value)[0][0]
                            time_index = np.where(self.box_c_ts[str(box_car)] == peak_value)[1][0] * (2 ** box_car)
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
        copy_array(self.block_N_2 , self.block_N_1)
        copy_array(self.block_N_1 , ispan.data)
#                self.block_N_1 = bf.copy(ispan.data, space = 'cuda')
#        else:
#            print('In block ' + str(self.n_iter) + '. Waiting for the next block')
            # Way 2:
#            dm_index = np.median(peak_values[final_beam,:,1])
#            final_time = np.median(peak_values[final_beam,:,2])
#            final_dm = dm_[int(dm_index)]
#            print('Way 2')
#            print(final_beam, final_box_car, final_dm, final_time)
#            print(datetime.now() - start)
        self.n_iter +=1           
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
#                self.tscrunched = odata[0,beam,:,:].copy(space = 'system')
#                peak_value = self.tscrunched.max()
#                peak_bxc_value.append(peak_value)
#            else:
#                bf.reduce(odata[0,beam,:,:], self.box_c[str(i)] , op = 'mean')
#                self.tscrunched = self.box_c[str(i)].copy(space = 'system')
#                peak_value = self.tscrunched.max()
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
#        self.n_iter += 1

def FDMT(iring, *args, **kwargs):
    return FDMT_block(iring, *args, **kwargs)
