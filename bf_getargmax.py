import bifrost as bf
import numpy as np

def getArgMax(data):
  maxrowdata = bf.ndarray(shape=(data.shape[0], 1), dtype=data.dtype, space='cuda')
  maxcoldata = bf.ndarray(shape=(1, data.shape[1]), dtype=data.dtype, space='cuda')
  bf.reduce(data, maxrowdata, op='max')
  bf.reduce(data, maxcoldata, op='max')
  localmaxrowdata = maxrowdata.copy(space='system')
  localmaxcoldata = maxcoldata.copy(space='system')
  return(np.argmax(localmaxrowdata), np.argmax(localmaxcoldata))
