from __future__ import absolute_import

from bifrost.pipeline import TransformBlock
import bifrost as bf

from copy import deepcopy
import numpy as np

SUBTRACTMEAN_KERNEL = """
// Compute o = i - m, where i.shape is (beam, fine_time, frequency) and m.shape is (beam, 1, frequency)
o(b,t,f) = i(b,t,f) - m(b, 0, f) 
"""

class NormaliseBlock(TransformBlock):
    # operation here would go away (operation will be "normalise")
    def __init__(self, iring, axis, op='sum', *args, **kwargs):
        super(NormaliseBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0
        self.specified_axis   = axis
    #    if self.n_iter % 8 == 0: # Every ~20 seconds?
            #####do it.
    #        self.cal_mv = True
        self.op = op
        # We might like to include another parameter which is how often to recalculate the mean and std dev
        # Specify new mean and new variance
    def define_valid_input_spaces(self):
        return ('cuda',)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
#        self.headershape = itensor["shape"]
        ohdr = deepcopy(ihdr)
        #print(ohdr['_tensor'])
#        otensor = ohdr['_tensor']
#        otensor['dtype'] = 'f32'
        
#        if itensor['dtype'] == 'cf32' and not self.op.startswith('pwr'): # This probably goes away as well
#            otensor['dtype'] = 'cf32'
#        if 'labels' in itensor and isinstance(self.specified_axis, str):
            # Look up axis by label
#            self.axis = itensor['labels'].index(self.specified_axis)
#        else:
#            self.axis = self.specified_axis
#        self.factor = otensor['shape'][self.axis]
#        print(otensor['shape'])

#        m_data = bf.ndarray(shape=(,4096), dtype='f32', space='system')

        # 1. Create a temporary data storage with shape (beam, 1, frequency) [This is for the mean]
        # 2. Create a temporary data storage with shape (beam, fine_time, frequency (This is for the input data - mean)
        # 3. Create a temporary data storage with shape (beam, 1, frequency) [This is for the variance]
        #otensor['scales'][self.axis][1] *= self.factor
        return ohdr
    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe
        idata, odata = ispan.data, ospan.data
        
#        bf.reduce(idata, odata, 'mean')
#        if self.n_iter  == 0 and self.op == 'var':
#            idata, odata = ispan.data, ospan.data
            # 1. Reduce into the first temporary data storage (bf.reduce)
#            bf.reduce(idata, odata, 'mean')
            # 2. Call the SUBTRACTMEAN KERNEL from the input (and using the mean) into the second temporary data storage
      #      bf.map(SUBTRACTMEAN_KERNEL, data={'i': idata, 'm': thefirsttempdata, 'o': thesecondtempdata},
#                                               axis_names=['b', 't', 'f'], shape=self.headershape)
            # 3. Call bf.reduce on the second temporary data storage into the third temporary data storage to get the variance
            # 4. Call a NORMALISE KERNEL (TO BE WRITTEN) which subtracts mean, divides by variance, multiples by target variance, adds target mean and puts result in output

#            omean = odata.copy(space='cuda')
#            print('omean: ' + str(omean.shape))
#                omean = 
#                odata = idata - omean
#                bf.reduce(data, sdata, op='pwrsum')
                #bf.reduce(, 'sum')

                #np.save(self.op, -omean)

                #bf.reduce(add_block, , 'sum')





            #elif self.n_iter  == 0:
            #    idata, odata = ispan.data, ospan.data
            #    bf.reduce(idata, odata, self.op)
                #np.save(self.op, odata)
            #print(self.n_iter)    
            #self.n_iter += 1
        # TODO: Support system space using Numpy
 #       #ishape = list(idata.shape)
        #ishape[self.axis] //= self.factor
        #ishape.insert(self.axis+1, self.factor)

def normalise(iring, axis, op='sum', *args, **kwargs):
    """Reduce data along an axis by factor using op.
    Args:
        iring (Ring or Block): Input data source.
        axis (int or str): The axis to reduce. Can be an integer index
                           or a string label.
        factor (int): The factor by which the axis should be reduced.
                      If None, the whole axis is reduced. Must divide
                      the size of the axis (or the gulp_size in the case
                      where the axis is the frame axis).
        op (str): The operation with which the data should be reduced.
                  One of: sum, mean, min, max, stderr [stderr=sum/sqrt(n)], 
                  pwrsum [magnitude squared sum], pwrmean, pwrmin, pwrmax, 
                  or pwrstderr.  Note:  min and max are not supported for 
                  complex valued data.
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.
    **Tensor semantics**::
        Input:  [..., N, ...], dtype = float, space = CUDA
        op = any
        Output: [..., N / factor, ...], dtype = f32, space = CUDA
        
        Input:  [..., N, ...], dtype = complex, space = CUDA
        op = 'sum', 'mean', 'stderr'
        Output: [..., N / factor, ...], dtype = cf32, space = CUDA
        
        Input:  [..., N, ...], dtype = complex, space = CUDA
        op = 'pwrsum', 'pwrmean', 'pwrmin', 'pwrmax', 'pwrstderr'
        Output: [..., N / factor, ...], dtype = f32, space = CUDA
    Returns:
        ReduceBlock: A new block instance.
    """
    print('Reached normalise function')
    return NormaliseBlock(iring, axis, op, *args, **kwargs)
