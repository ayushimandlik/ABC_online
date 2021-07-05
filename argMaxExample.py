#!/home/npsr/miniconda2/envs/ABC_bifrost/bin/python
import numpy as np
import bifrost as bf
from bf_getargmax import getArgMax
d = np.zeros(10000).reshape(100,100)
d[5][7] = 1
d[10][2] = 1
#data =  bf.ndarray([[0,1,2,3],[4,5,6,7],[9,0,0,0],[1,1,1,3]], space='cuda', dtype=np.float32)
data = bf.ndarray(d, space='cuda', dtype=np.float32)
# Maximum value is 9, at position (2,0)

print("Shape of input data is", data.shape)
print("Maximum value is at position", getArgMax(data))
