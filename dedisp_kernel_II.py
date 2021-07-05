DEDISPBOX1_KERNEL = """
// All inputs have axes (beam, frequency, time)
// input i (the data) has shape 5, 512, 3x8192
// time delay td (the frequency-dependent offset to the first time sample to select) has shape (1, 512, 1)
// Compute o = i shifted by td and averaged by a factor of 1
// The shape of the output is (5, 256, 256)
// we have defined the axis names as b, f, t
o(b, f, t) = i(b, f*2, t + td(1, f*2, 1));
o(b, f, t) += i(b, f*2+1, t + td(1, f*2+1, 1))
"""

DEDISPBOX2_KERNEL = """
// All inputs have axes (beam, frequency, time)
// input i (the data) has shape 5, 512, 3x8192
// time delay td (the frequency-dependent offset to the first time sample to select) has shape (1, 512, 1)
// Compute o = i shifted by td and averaged by a factor of 1
// The shape of the output is (5, 256, 256)
// we have defined the axis names as b, t, f
o(b, f, t) = i(b, f*2, t*2 + td(1, f*2, 1));
o(b, f, t) += i(b, f*2, t*2 + td(1, f*2, 1) + 1);
o(b, f, t) += i(b, f*2+1, t*2 + td(1, f*2+1, 1));
o(b, f, t) += i(b, f*2+1, t*2 + td(1, f*2+1, 1) + 1)
"""
import numpy as np
import bifrost as bf
import matplotlib.pyplot as plt

#data = numpy.random.randn(10, 20, 30, 40)
#data = data.astype(numpy.float32)
#tdata = data.transpose(1,3,2,0).copy()
#print('numpy:', data[1,3,5,7], '->', tdata[3,7,5,1])
#
#data = bifrost.ndarray(data, space='cuda')
#tdata = bifrost.ndarray(shape=[data.shape[v] for v in (1,3,2,0)],
#                        dtype=data.dtype, space='cuda')
#bifrost.transpose.transpose(tdata, data, axes=(1,3,2,0))
#data2 = data.copy(space='system')
#tdata2 = tdata.copy(space='system')
#print('bifrost:', data2[1,3,5,7], '->', tdata2[3,7,5,1])
#
#
#DM = 1.25
#time = numpy.linspace(0, 2, 1000)
#freq = numpy.linspace(50e6, 70e6, 200)
#data = numpy.random.randn(freq.size, time.size)
#data[:,100] += 10.0
#data[:,101] += 8.0
#data[:,102] += 4.0
#data[:,103] += 1.0
#td = numpy.zeros((freq.size), dtype = int)
#for i in range(freq.size):
#    delay = 4.15e-3 * DM * ((freq[i]/1e9)**-2 - (freq[-1]/1e9)**-2)
#    delay = int(round(delay / (time[1] - time[0])))
#    td[i] = delay
#    data[i,:] = numpy.roll(data[i,:], delay)
#
#print(td)
#print(data.shape)
#td = td.reshape(1, 200, 1)
#data = data.reshape(1, 200, 1000).astype(numpy.float32)
#i = bifrost.ndarray(data, space='cuda')
#td = bifrost.ndarray(td, space='cuda')
#print(data.shape)
#print(td.shape)
#ddata = bifrost.ndarray(shape=(time.size, time.size),
#                        dtype=data.dtype, space='cuda')

#o = bf.ndarray(shape=(5, 256, 3*8192), dtype=np.float32, space='cuda')
#o = bifrost.ndarray(shape=(1, 200, 1000), dtype=numpy.float32, space='cuda')
#i = bf.ndarray(shape=(1, 200, 1000), dtype=np.float32, space='cuda')
o = bf.ndarray(shape=(5, 256, 256), dtype=np.float32, space='cuda')
i = bf.ndarray(shape=(5, 512, 3*8192), dtype=np.float32, space='cuda')
td = bf.ndarray(shape=(1, 512, 1), dtype=np.float32, space='cuda')
x = np.random.normal(0, 1, (5* 512* 3*8192)).reshape(5, 512, 3*8192)
t = np.arange(512).reshape(1, 512, 1)
for m in range(8192):
  for j in range(512):
    for k in range(5):
      if m == j:
        x[k, j, m * 47] = 20
#i = bf.ndarray(x, space = 'cuda')
#bf.map(DEDISPBOX_KERNEL, data={'o': o, 'i': i, 'td': td}, axis_names = ['b', 'f', 't'], shape=(5, 256, 8192*3))
bf.map(DEDISPBOX2_KERNEL, data={'o': o, 'i': i, 'td': td}, axis_names = ['b', 'f', 't'], shape=(5, 256, 256))
plt.figure()
#inp = data[0,:,:].copy(space = 'system')
#print(i.shape)
#plt.imshow(data[0, :, :], aspect = 'auto')
#plt.savefig('input_ii.png')
#print(o)
print(o.shape)
#plt.figure()
out = o.copy(space = 'system')
#out = td.copy(space = 'system')
#plt.imshow(out, aspect = 'auto')
#plt.savefig('output_ii.png')

