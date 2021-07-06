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

DEDISP_KERNEL = """
// All inputs have axes (beam, frequency, time)
// input i (the data) has shape 5, 512, 3x8192
// time delay td (the frequency-dependent offset to the first time sample to select) has shape (1, 512, 1)
// Compute o = i shifted by td and averaged by a factor of 1
// The shape of the output is (5, 256, 3x8192)
// we have defined the axis names as t, b, ft, f
o(1, b, f, t) = i(1, b, f, t + td(1, 1, f, 1));
"""

class FDMT_block(bf.pipeline.TransformBlock):
    def __init__(self, iring, *args, **kwargs):
        super(FDMT_block, self).__init__(iring, *args, **kwargs)
        self.kdm       = 4.148741601e3 # MHz**2 cm**3 s / pc
        self.dm_units  = 'pc cm^-3'
        self.timechunksize = 8192
        self.n_iter = 0
        self.num_beam_blocks = 3 

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        refdm = 0
        dmstep = 1
        ohdr['name'] = 'FDMT_block'
        ohdr['_tensor']['dtype'] = 'f32'

#        ohdr['_tensor']['shape'][0] = self.num_beam_blocks
#        ohdr['_tensor']['shape'][1] = 2
#        ohdr['_tensor']['shape'][2] = 256
#        ohdr['_tensor']['shape'][3] = 256
#        ohdr['_tensor']['shape'][4] = 5
#        ohdr['_tensor']['shape'][-1]  = 256
        ohdr['_tensor']['shape'][-1]  = self.timechunksize
        ohdr['_tensor']['shape'][-2]  = self.timechunksize
        ohdr['_tensor']['labels'][-2] = 'dispersion'
        ohdr['_tensor']['scales'][-2] = (refdm, dmstep) # These need to be calculated properly
        ohdr['_tensor']['units'][-2]  = self.dm_units
        self.reffreq = 806.25
        self.freq_step = 100./1024
        self.f = bf.fdmt.Fdmt()
        ohdr['_tensor']['shape'][1]  = self.num_beam_blocks # for the number of beams across which to search for the pulse.
        self.f.init(512, self.timechunksize, self.reffreq, self.freq_step)

#        self.dm_array = bf.ndarray(shape = (self.timechunksize, 1), dtype = np.float32, space = 'system')
#        self.time_array = bf.ndarray(shape = (1, self.timechunksize), dtype = np.float32, space = 'system')
# works in cuda space too, but trying to see if it can be used with system memory so less cuda space is used.
        self.block_full = bf.ndarray(shape = (1, 24, 512, 3 * self.timechunksize), dtype = np.float32, space = 'system')
        self.td = bf.ndarray(shape = (1, 1, 512, 1), dtype = int, space = 'cuda')
        self.beam_data = bf.ndarray(shape = (1, 5, 512, 3 * self.timechunksize), dtype = np.float32, space = 'cuda')
        self.idata = bf.ndarray(shape = (3, 512, 8192), dtype = np.float32, space = 'cuda')
        #self.data = bf.ndarray(shape = (1, 5, 512, 3 * self.timechunksize), dtype = np.float32, space = 'cuda')

#        self.block_N_1 = bf.ndarray(shape = (1, 24, 512, 8192), dtype = np.float32, space = 'system') 
#        self.block_N_2 = bf.ndarray(shape = (1, 24, 512, 8192), dtype = np.float32, space = 'system') 
        self.fdmt_block = bf.ndarray(shape = (1, self.num_beam_blocks * 3, self.timechunksize, self.timechunksize), dtype = np.float32, space = 'cuda')

        self.box_c_dm = bf.ndarray(shape = (self.timechunksize, 1), dtype = np.float32, space = 'cuda')
        self.dedispersed = bf.ndarray(shape = (1, 5, 512, 128), dtype = np.float32, space = 'cuda')
        self.candidate = bf.ndarray(shape = (1, 3, 5, 256, 128), dtype = np.float32, space = 'cuda')
        
        self.bx_0 = bf.ndarray(shape = (self.timechunksize, self.timechunksize), dtype = np.float32, space = 'cuda')
        self.bx_1 = bf.ndarray(shape = (self.timechunksize, self.timechunksize // 2), dtype = np.float32, space = 'cuda')
        self.bx_2 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 4), dtype = np.float32, space = 'cuda')
        self.bx_3 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 8), dtype = np.float32, space = 'cuda')
        self.bx_4 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 16), dtype = np.float32, space = 'cuda')
        self.bx_5 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 32), dtype = np.float32, space = 'cuda')
        self.bx_6 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 64), dtype = np.float32, space = 'cuda')
        self.bx_7 = bf.ndarray(shape = (self.timechunksize , self.timechunksize // 128), dtype = np.float32, space = 'cuda')
        self.box_c = {'0': self.bx_0, '1': self.bx_1, '2': self.bx_2, '3': self.bx_3, '4': self.bx_4, '5': self.bx_5, '6': self.bx_6, '7': self.bx_7}

        self.bx_0_ts = bf.ndarray(shape = (1 ,self.timechunksize), dtype = np.float32, space = 'cuda')
        self.bx_1_ts = bf.ndarray(shape = (1 ,self.timechunksize // 2), dtype = np.float32, space = 'cuda')
        self.bx_2_ts = bf.ndarray(shape = (1 ,self.timechunksize // 4), dtype = np.float32, space = 'cuda')
        self.bx_3_ts = bf.ndarray(shape = (1 ,self.timechunksize // 8), dtype = np.float32, space = 'cuda')
        self.bx_4_ts = bf.ndarray(shape = (1 ,self.timechunksize // 16), dtype = np.float32, space = 'cuda')
        self.bx_5_ts = bf.ndarray(shape = (1 ,self.timechunksize // 32), dtype = np.float32, space = 'cuda')
        self.bx_6_ts = bf.ndarray(shape = (1 ,self.timechunksize // 64), dtype = np.float32, space = 'cuda')
        self.bx_7_ts = bf.ndarray(shape = (1 ,self.timechunksize // 128), dtype = np.float32, space = 'cuda')
        self.box_c_ts = {'0': self.bx_0_ts, '1': self.bx_1_ts, '2': self.bx_2_ts, '3': self.bx_3_ts, '4': self.bx_4_ts, '5': self.bx_5_ts, '6': self.bx_6_ts, '7': self.bx_7_ts}
        return ohdr
        
    def on_data(self, ispan, ospan):
        if self.n_iter >= 2:
            self.predictions = np.load('Direct_classifier_logs_' + str(self.n_iter - 1) + '.npy')
            self.furby_predictions = self.predictions[:, 1, :]
            beam_blocks = [] 
            #### TODO will be replaced by a condition which will decide on whether any of this should happen. ####
            for beam in sorted(np.ravel(self.furby_predictions), reverse = True)[:self.num_beam_blocks]:
                beam_blocks.append(np.concatenate(np.argwhere( self.furby_predictions == beam ))[::3]) # Set the threshold values here
    
            in_nframe  = ispan.nframe
            odata = ospan.data
            out_nframe = in_nframe
            freq = np.arange(806.25, 856.25, 0.09765625)*1e6
    
            ### peak_values is array containing information about peak values in terms of (beam, boxcar, time_index, dm_index)
            for ob_index, beam_bl in enumerate(beam_blocks):
                # ob_index: output beam index.
                # beam_bl: the index of beam in the whole block with all beams.
                start = datetime.now()
                peak_values = np.zeros((3, 8, 3), dtype = np.float32)
                print(self.idata.shape)
                print(self.block_full[0, (3 * beam_bl[0]): (3 * beam_bl[0]) + 3, :, self.timechunksize : self.timechunksize * 2 ].shape)
                copy_array(self.idata, self.block_full[0, (3 * beam_bl[0]): (3 * beam_bl[0]) + 3, :, self.timechunksize : self.timechunksize * 2 ])
                for beam_num in range(3):
#                    self.f.execute(self.block_full[0, (3 * beam_bl[0]) + beam_num, :, self.timechunksize : self.timechunksize * 2 ], 
#                        self.fdmt_block[0, (3 * ob_index) + beam_num, :, :], negative_delays = True) 
                    self.f.execute(self.idata[beam_num, :, : ], self.fdmt_block[0, (3 * ob_index) + beam_num, :, :], negative_delays = True) 
                    for box_car in range(8):
                        if box_car == 0:
                            bf.reduce(self.fdmt_block[0, (3 * ob_index) + beam_num, :, :], self.box_c_dm, op = 'max')
                            # barfs when run twice. WHY?!
                            dm_array = self.box_c_dm.copy(space = 'system')
                            #copy_array(self.dm_array ,self.box_c_dm)
                            peak_values[beam_num, box_car, 0] = dm_array.max()
                            peak_values[beam_num, box_car, 1] = np.argmax(dm_array)
                            bf.reduce(self.fdmt_block[0, (3 * ob_index) + beam_num, :, :], self.box_c_ts[str(box_car)], op = 'max')
                            time_array = self.box_c_ts[str(box_car)].copy(space = 'system')
                            peak_values[beam_num, box_car, 2] = np.argmax(time_array)
                        else:
                            bf.reduce(self.fdmt_block[0, (3 * ob_index) + beam_num, :, :], self.box_c[str(box_car)] , op = 'mean')
                            bf.reduce(self.box_c[str(box_car)], self.box_c_dm, op = 'max')
                            dm_array = self.box_c_dm.copy(space = 'system')
#                            copy_array(dm_array, self.box_c_dm)
                            peak_values[beam_num, box_car, 0] = dm_array.max()
                            peak_values[beam_num, box_car, 1] = np.argmax(dm_array)
                            bf.reduce(self.box_c[str(box_car)], self.box_c_ts[str(box_car)], op = 'max')
#                            self.time_array = np.zeros((1, 8192))
                            time_array = self.box_c_ts[str(box_car)].copy(space = 'system')
                            peak_values[beam_num, box_car, 2] = np.argmax(time_array)
            
                beam_ = np.where(peak_values == peak_values[:,:,0].max())[0][0]
                print(peak_values)
                np.save('peak_values', peak_values)
                beam = (3 * beam_bl[0]) + beam_
                final_box_car = np.where(peak_values == peak_values[:,:,0].max())[1][0]
                
                time = np.arange(0, (0.00032768 * self.timechunksize), 0.00032768)
                tint = time[1] - time[0]
                dm_ = np.arange(time.size)*1.0
                dm_ *= tint / 4.15e-3 / ((freq[0]/1e9)**-2 - (freq[-1]/1e9)**-2)
            
                # Way 1:
                dm_index = peak_values[beam_, final_box_car, 1] 
                tstart = peak_values[beam_, final_box_car, 2]
                print(tstart)
                dm = dm_[int(dm_index)]

                width = 2 ** final_box_car
                tstop_s = int(tstart + self.timechunksize + ( width * 64 ))
                tstart_s = int(tstart + self.timechunksize - ( width * 64 ))
                print(tstart_s, tstop_s)
                if tstop_s > 3 * self.timechunksize:
                    tstop_s = 3 * self.timechunksize
                # Should not be used.
                if tstart_s < 0:
                    tstop_s = 0

                print(beam, final_box_car, dm, tstart)
                print(datetime.now() - start)

                delay_samples = np.zeros((1, 1, 512, 1), dtype = int)
                for i in range(512):
                    delay = 4.15e-3 * dm * (( freq[i] / 1e9 )**-2 - ( freq[-1] / 1e9 )**-2)
                    delay_samples[0, 0, i, 0] = int(round(delay / (time[1] - time[0])))

                copy_array(self.td, delay_samples)

                if beam > 1 and beam < 21:
                    print(beam - 2, beam + 3)
                    print(self.block_full[0, (beam - 2):(beam + 3), :, :].shape)
                    copy_array(self.beam_data[0, :, :, :], self.block_full[0, (beam - 2):(beam + 3), :, :])

                print(self.beam_data.shape)
                print(self.td.shape)
                print(self.beam_data[:, :, :, tstart_s: tstop_s].shape)

# Change in Kernel, as the decimation for time should depend on the width determined. Reducing after dedispersion makes sense as delay depends on 512 channels. Can be changed.
                bf.map(DEDISP_KERNEL, data={'o': self.beam_data, 'i': self.beam_data, 'td': self.td}, axis_names = ['t', 'b', 'f', 'ft'], shape = (1, 5, 512, 3 * self.timechunksize))
                print(self.beam_data[:, :, :, tstart_s: tstop_s].shape)
                if width > 1:
                    bf.reduce(self.beam_data[:, :, :, tstart_s: tstop_s], self.dedispersed, op = 'mean')
                else:
                    copy_array(self.dedispersed, self.beam_data[:, :, :, tstart_s: tstop_s])
                print(self.candidate[0, ob_index, :, :, :].shape, self.dedispersed[0, :, :, :].shape)
                bf.reduce(self.dedispersed[0, :, :, :], self.candidate[0, ob_index, :, :, :], op = 'mean')

                # This will return the dedispersed block. dmt block still needs to be decided upon. 

        copy_array(self.block_full[:, :, :, 2*self.timechunksize:] , self.block_full[:, :, :, self.timechunksize : self.timechunksize * 2])
        copy_array(self.block_full[:, :, :, self.timechunksize : self.timechunksize * 2] , ispan.data)
        self.n_iter +=1           

def FDMT(iring, *args, **kwargs):
    return FDMT_block(iring, *args, **kwargs)
