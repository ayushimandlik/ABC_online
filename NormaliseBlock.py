from __future__ import absolute_import

from bifrost.pipeline import TransformBlock
import bifrost as bf

from copy import deepcopy
import numpy as np

SUBTRACTMEAN_KERNEL = """
// Compute o = i - m, where i.shape is (time, beam, fine_time, frequency) and m.shape is (time, beam, 1, frequency)
o(t, b, ft, f) = i(t, b, ft, f) - m(1, b, 1, f) 
"""

NORMALISE_KERNEL = """
// Compute o = ((i - m) * r)/ v, where o.shape is (time, beam, fine_time, frequency) and m.shape is (time, beam, 1, frequency), s.shape is (1, 1, 1, 1)
o(t, b, ft, f) = ((i(t, b, ft, f) - m(1, b, 1, f)) * r) / v(1, b, 1, f)
"""

class NormaliseBlock(TransformBlock):
    # operation here would go away (operation will be "normalise")
    def __init__(self, iring, *args, **kwargs):
        super(NormaliseBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0
    #    if self.n_iter % 8 == 0: # Every ~20 seconds?
    #        self.cal_mv = True
        # We might like to include another parameter which is how often to recalculate the mean and std dev
        # Specify new mean and new variance

    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        self.headershape = itensor["shape"]
        ohdr = deepcopy(ihdr)
        otensor = ohdr['_tensor']
        otensor['dtype'] = 'f32'
        
        self.time_shape = itensor['shape'][itensor['labels'].index('fine_time')]
        self.beam_shape = itensor['shape'][itensor['labels'].index('beam')]
        self.freq_shape = itensor['shape'][itensor['labels'].index('freq')]

        self.m_data = bf.ndarray(shape=(1, self.beam_shape, 1, self.freq_shape), dtype='f32', space='cuda')
        self.v_data = bf.ndarray(shape=(1, self.beam_shape, 1, self.freq_shape), dtype='f32', space='cuda')
        self.s_data = bf.ndarray(shape=(1, self.beam_shape, self.time_shape, self.freq_shape), dtype='f32', space='cuda')
        return ohdr

    def on_data(self, ispan, ospan):
        idata, odata = ispan.data, ospan.data
        bf.reduce(idata, self.m_data, 'mean')
        bf.map(SUBTRACTMEAN_KERNEL, data={'i': idata, 'm': self.m_data, 'o': self.s_data}, axis_names=['t', 'b', 'ft', 'f'], shape=(1, self.beam_shape, self.time_shape, self.freq_shape))
        bf.reduce(self.s_data, self.v_data, op='pwrsum')
        bf.map(NORMALISE_KERNEL, data={'m': self.m_data, 'v': self.v_data, 'i': idata, 'o': odata, 'r': self.time_shape}, axis_names=['t', 'b', 'ft', 'f'], shape=(1, self.beam_shape, self.time_shape, self.freq_shape))
        self.n_iter += 1

def normalise(iring, *args, **kwargs):
    return NormaliseBlock(iring, *args, **kwargs)
