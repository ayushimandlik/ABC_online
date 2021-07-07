import bifrost as bf
from copy import deepcopy
from datetime import datetime
from pprint import pprint
import time as ptime
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
from bifrost.ndarray import copy_array

DEDISP_KERNEL = """
// All inputs have axes (beam, frequency, time)
// input i (the data) has shape 5, 512, 3x8192
// time delay td (the frequency-dependent offset to the first time sample to select) has shape (1, 512, 1)
// Compute o = i shifted by td and averaged by a factor of 1
// The shape of the output is (5, 256, 3x8192)
// we have defined the axis names as t, b, ft, f
o(b, f, ft) = i(b, f, ft + td(1, f, 1));
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

        #print(ohdr)
        refdm = 0
        dmstep = 1
        ohdr['name'] = 'FDMT_block'
        ohdr['_tensor']['dtype'] = 'f32'
        ohdr['_tensor']['labels'] = ['time', 'beam_block', 'ft_dmt', 'beam', 'freq', 'time']
        ohdr['_tensor']['shape']  = [-1, 3, 2, 5, 256, 128]
        ohdr['_tensor']['units'][-2]  = ['', '', 'self.dm_units', '', 'MHz', 's']
#        ohdr['_tensor']['scales'][-2] = (refdm, dmstep) # These need to be calculated properly
        self.reffreq = 806.25
        self.freq_step = 100./1024
        self.f = bf.fdmt.Fdmt()
        self.dmt = bf.fdmt.Fdmt()
        ohdr['_tensor']['shape'][1]  = self.num_beam_blocks # for the number of beams across which to search for the pulse.
        self.f.init(512, self.timechunksize, self.reffreq, self.freq_step)
        self.dmt.init(512, 16384, self.reffreq, self.freq_step)

        #self.dm_values = bf.ndarray(shape = (self.timechunksize, 1), dtype = np.float32, space = 'cuda')
        self.block_full = bf.ndarray(shape = (1, 24, 512, 3 * self.timechunksize), dtype = np.float32, space = 'system')
        self.td = bf.ndarray(shape = (1, 512, 1), dtype = np.uint16, space = 'cuda')
#        self.beam_data = bf.ndarray(shape = (1, 5, 512, 3 * self.timechunksize), dtype = np.float32, space = 'cuda')
        #self.data = bf.ndarray(shape = (1, 5, 512, 3 * self.timechunksize), dtype = np.float32, space = 'cuda')

#        self.block_N_1 = bf.ndarray(shape = (1, 24, 512, 8192), dtype = np.float32, space = 'system') 
#        self.block_N_2 = bf.ndarray(shape = (1, 24, 512, 8192), dtype = np.float32, space = 'system') 
        self.fdmt_block = bf.ndarray(shape = (1, 3, self.timechunksize, self.timechunksize), dtype = np.float32, space = 'cuda')
        self.time_block = bf.ndarray(shape = (512, 16384), dtype = np.float32, space = 'cuda')
        self.idata = bf.ndarray(shape = (512, 8192), dtype = np.float32, space = 'cuda')

        self.box_c_dm = bf.ndarray(shape = (self.timechunksize, 1), dtype = np.float32, space = 'cuda')
        self.dedispersed = bf.ndarray(shape = (5, 512, 3 * self.timechunksize), dtype = np.float32, space = 'cuda')
        self.dmtime = bf.ndarray(shape = (16384, 16384), dtype = np.float32, space = 'cuda')
        
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
        
        odata = ospan.data
        if self.n_iter >= 2:
        #    odata = ospan.data
            self.predictions = np.load('Direct_classifier_logs_' + str(self.n_iter - 1) + '.npy')
            self.furby_predictions = self.predictions[:, 1, :]
            beam_blocks = [] 
            for beam in sorted(np.ravel(self.furby_predictions), reverse = True)[:self.num_beam_blocks]:
                beam_blocks.append(np.concatenate(np.argwhere( self.furby_predictions == beam ))[::2]) # Set the threshold values here
    
            in_nframe  = ispan.nframe
            print(ospan.data.shape)
            out_nframe = in_nframe
            freq = np.arange(806.25, 856.25, 0.09765625)*1e6
            print(beam_blocks)
    
            ### peak_values is array containing information about peak values in terms of (beam, boxcar, time_index, dm_index)
            for ob_index, beam_bl in enumerate(beam_blocks):
                # ob_index: output beam index.
                # beam_bl: the index of beam in the whole block with all beams.
                start = datetime.now()
                peak_values = np.zeros((3, 8, 3), dtype = np.float32)
    
                for beam_num in range(3):
                    copy_array(self.idata, self.block_full[0, (3 * beam_bl[0]) + beam_num, :, self.timechunksize : self.timechunksize * 2])
                    self.f.execute(self.idata, self.fdmt_block[0, beam_num, :, :], negative_delays = True)
                    for box_car in range(8):
                        if box_car == 0:
                            bf.reduce(self.fdmt_block[0, beam_num, :, :], self.box_c_dm, op = 'max')
                            dm_array = self.box_c_dm.copy(space = 'system')
                            peak_values[beam_num, box_car, 0] = dm_array.max()
                            peak_values[beam_num, box_car, 1] = np.argmax(dm_array)
                            bf.reduce(self.fdmt_block[0, beam_num, :, :], self.box_c_ts[str(box_car)], op = 'max')
                            time_array = self.box_c_ts[str(box_car)].copy(space = 'system')
                            peak_values[beam_num, box_car, 2] = np.argmax(time_array)
            
                        else:
                            bf.reduce(self.fdmt_block[0, beam_num, :, :], self.box_c[str(box_car)] , op = 'mean')
                            bf.reduce(self.box_c[str(box_car)], self.box_c_dm, op = 'max')
                            dm_array = self.box_c_dm.copy(space = 'system')
                            peak_values[beam_num, box_car, 0] = dm_array.max()
                            peak_values[beam_num, box_car, 1] = np.argmax(dm_array)
                            bf.reduce(self.box_c[str(box_car)], self.box_c_ts[str(box_car)], op = 'max')
                            time_array = self.box_c_ts[str(box_car)].copy(space = 'system')
                            peak_values[beam_num, box_car, 2] = np.argmax(time_array)
                np.save('peak_values', peak_values)
            
                beam_ = np.where(peak_values == peak_values[:,:,0].max())[0][0]
                beam = (3 * beam_bl[0]) + beam_
                final_box_car = np.where(peak_values == peak_values[:,:,0].max())[1][0]
                
                time = np.arange(0, (0.00032768 * self.timechunksize), 0.00032768)
                tint = time[1] - time[0]
                dm_ = np.arange(time.size)*1.0
                dm_ *= tint / 4.15e-3 / ((freq[0]/1e9)**-2 - (freq[-1]/1e9)**-2)

            
                # Way 1:
                dm_index = peak_values[beam_, final_box_car, 1] 
                tstart = peak_values[beam_, final_box_car, 2]
                dm = dm_[int(dm_index)]

                width = 2 ** final_box_car
                tstop_s = int(tstart + self.timechunksize + ( width * 64 ))
                tstart_s = int(tstart + self.timechunksize - ( width * 64 ))
                tstart_dmt = int(tstart)
                tstop_dmt = int(tstart + 16384)
                tstart_dmt2 = int(8192 - ( width * 64 ))
                tstop_dmt2 = int(8192 + ( width * 64 ))


                if tstop_s > 3 * self.timechunksize:
                    tstop_s = 3 * self.timechunksize
                # Should not be used.
                if tstart_s < 0:
                    tstop_s = 0

                print(beam, final_box_car, dm, tstart)
                print(datetime.now() - start)

                delay_samples = np.zeros((1, 512, 1), dtype = np.uint16)
                for i in range(512):
                    delay = 4.15e-3 * dm * (( freq[i] / 1e9 )**-2 - ( freq[-1] / 1e9 )**-2)
                    delay_samples[0, i, 0] = int(round(delay / (time[1] - time[0])))

                copy_array(self.td, delay_samples)
                beam = 15

                if beam > 1 and beam < 21:
                    copy_array(self.dedispersed, self.block_full[0, (beam - 2):(beam + 3), :, :])
                    bf.map(DEDISP_KERNEL, data={'o': self.dedispersed, 'i': self.dedispersed, 'td': self.td}, axis_names = ['b', 'f', 'ft'], shape = (5, 512, 3 * self.timechunksize))
                    print(self.dedispersed[:, :, tstart_s: tstop_s].shape)
                    print(odata[0, ob_index, 0, :, :, :].shape)
                    bf.reduce(self.dedispersed[:, :, tstart_s: tstop_s], odata[0, ob_index, 0, :, :, :], op = 'mean')
                    #bf.reduce(self.dedispersed[:, :, tstart_s: tstop_s], self.candidate[0, ob_index, :, :, :], op = 'mean')

                    for adj_beam in range(5):
                        #self.dmtime: 16384, 16384.
                        print(self.block_full.shape)
                        #self.time_block: 512, 16384.
                        print(self.block_full[0, (beam + adj_beam - 2), :, tstart_dmt:tstop_dmt].shape)
                        print(self.time_block.shape)
                        copy_array(self.time_block, self.block_full[0, (beam + adj_beam - 2), :, tstart_dmt:tstop_dmt])


                        self.dmt.execute(self.time_block, self.dmtime, negative_delays = True)
                        #self.dmt.execute(data, self.dmtime, negative_delays = True)
                        print(int(dm_index) - 128)

                        print(tstart_dmt2, tstop_dmt2)
                        if dm_index < 128:
                            # For dm < 58.1.
                            print('here')
                            print(self.dmtime[0: 256, tstart_dmt2:tstop_dmt2].shape)
                            print(odata[0, ob_index, 1, adj_beam, :, :].shape)
                            if width == 1:
                                copy_array(odata[0, ob_index, 1, adj_beam, :, :], self.dmtime[0: 256, tstart_dmt2:tstop_dmt2])
                            else:
                                bf.reduce(self.dmtime[0: 256, tstart_dmt2:tstop_dmt2], odata[0, ob_index, 1, adj_beam, :, :], op = 'mean')
                        else:
                            print(self.dmtime[int(dm_index) - 128: int(dm_index) + 128, tstart_dmt2:tstop_dmt2].shape)
                            if width == 1:
                                copy_array(odata[0, ob_index, 1, adj_beam, :, :], self.dmtime[int(dm_index) - 128: int(dm_index) + 128, tstart_dmt2:tstop_dmt2])
                            else:
                                bf.reduce(self.dmtime[int(dm_index) - 128: int(dm_index) + 128, tstart_dmt2:tstop_dmt2], odata[0, ob_index, 1, :, :, :], op = 'mean')
        np.save('odata', odata)
                        
        copy_array(self.block_full[:, :, :, 2*self.timechunksize:] , self.block_full[:, :, :, self.timechunksize : self.timechunksize * 2])
        copy_array(self.block_full[:, :, :, self.timechunksize : self.timechunksize * 2] , ispan.data)
#        ptime.sleep(3)
        self.n_iter +=1           

def FDMT(iring, *args, **kwargs):
    return FDMT_block(iring, *args, **kwargs)
