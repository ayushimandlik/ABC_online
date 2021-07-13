#!/home/npsr/miniconda2/envs/ABC_bifrost/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import gridspec

class ReadDada():
    def __init__(self, args):
        self.filename =  args.file_name
        self.get_header()
        self.save = args.save
        if int(self.header['NCHAN']) == 64:
            self.corner_turned = False
        elif int(self.header['NCHAN']) == 512:
            self.corner_turned = True
        self.dm = args.dedispersion_dm
        self.beam = args.beam
        self.nsamps = args.num_samples
        self.fscrunch = args.fscrunch
        self.tscrunch = args.tscrunch
        self.save = args.save
        self.tstart = args.start_sample
        self.tstop = args.end_sample
        self.get_freqs()

    def get_hdr_size(self):
        dada = open(self.filename, 'rb')
        dada.seek(0)
        dada_ = dada.read(4096)
        for i in dada_.split(b'\n'):
            key = i.split()[0]
            if key == b'HDR_SIZE':
                header_size = int(i.split(b'HDR_SIZE')[-1])
                break
        return header_size
    
    def get_header(self):
        self.header_size = self.get_hdr_size()
        dada = open(self.filename, 'rb')
        dada.seek(0)
        hdr = dada.read(self.header_size)
        header = {}
        for entry in hdr.split(b'\n'):
            try:
                header[entry.split()[0].decode('ASCII')]= entry.split()[1].decode('ASCII')
            except:
                pass
        self.header = header
    
    def get_data(self):
        dada = open(self.filename, 'r')
        dada.seek(self.header_size)
        nbit = int(self.header['NBIT'])
        if nbit == 8:
            dtype = 'uint8'
        elif nbit== 32:
            dtype = 'float32'
        data = np.fromfile(dada, count=-1, dtype=dtype)
        nbeam = int(self.header['NBEAM'])
        resolution = int(self.header['RESOLUTION'])
        nchan = int(self.header['NCHAN'])    
        time_per_block = int(resolution / (nchan * nbeam * nbit / 8))
        if self.header['ORDER'] == 'STF':
            data = data.reshape(-1, nbeam, time_per_block, nchan)
            data = np.moveaxis(data, -1, -2)
        if self.header['ORDER'] == 'SFT':
            data = data.reshape(-1, nbeam, nchan, time_per_block)
        block_sft = data[0]
        for block in range(data.shape[0] - 1):
            block_sft = np.concatenate((block_sft, data[block + 1]), axis = -1)
        if self.tstart == None:
            self.data = block_sft[self.beam, :, :self.nsamps]
        if self.tstart !=None:
            self.data = block_sft[self.beam, :, self.tstart:self.tstop]

    def get_freqs(self):
        chw = float(self.header['CHAN_BW'])
        central_freq = float(self.header['FREQ'])
        nchan = int(self.header['NCHAN'])
        f_low = central_freq - ((nchan // 2) * abs(chw))  - (chw/2)
        f_high = central_freq + ((nchan // 2) * abs(chw))  - (chw/2)
        self.freqs = np.arange(f_low, f_high, chw)*1e6
    
    def dedisperse(self):
        # Data has to be in the (freq, time) format for this function. v.i
        # output is also (f, t).
        tsamp = float(self.header['TSAMP']) *1e-6
        delay_time = 4148808.0 * self.dm * ( (1.0/(self.freqs[0]/ 1e6))**2 - (1.0/(self.freqs/1e6))**2) / 1000.0 
        #delay_samp = np.round(delay_time / tsamp).astype('int64')
        delay_samp = np.round(delay_time / tsamp).astype('int64')[::-1]
        data = self.data.T
        dedispersed = np.copy(data)
        for ii in range(int(self.header['NCHAN'])):
            dedispersed[:, ii] = np.concatenate([data[-delay_samp[ii]:, ii], data[:-delay_samp[ii], ii]])
        self.data = dedispersed.T
    
    def tdecimate(self):
        # Data has to be in the (freq, time) format for this function.
        # output is in in (freq, time) - easier to plot.
        data = self.data
        tddata = data.reshape(data.shape[0], (data.shape[1] // self.tscrunch), self.tscrunch).mean(-1)
        self.data = tddata


    def fdecimate(self):
        # Data has to be in the (freq, time) format for this function.
        # output is in in (freq, time) - easier to plot.
        data = self.data.T
        fddata = data.reshape(data.shape[0], (data.shape[1] // self.fscrunch), self.fscrunch).mean(-1)
        self.data = fddata.T

    def Plot(self):
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(2, 2, width_ratios=[6, 1], height_ratios=[1, 6])
        if self.tscrunch != None:
            times = np.arange(self.nsamps // self.tscrunch) * float(self.header['TSAMP']) * self.tscrunch * 1e-6 
        else:
            times = np.arange(self.nsamps) * float(self.header['TSAMP']) * 1e-6 
        ax1 = fig.add_subplot(gs[0, 0])
#        ax1.plot(self.data.sum(axis = 0))
        ax1.plot(times, self.data.sum(axis = 0))
        ax1.set_xlim(np.min(times),np.max(times))
        #ax1.set_xticks(times)
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.imshow(self.data, aspect = 'auto', extent = [ times[0], times[-1], self.freqs[0] / 1e6, self.freqs[-1] / 1e6])
        ax2.set_xlabel('Time in s')
        ax2.set_ylabel('Frequency in MHz')
        ax3 = fig.add_subplot(gs[1, 1])
        y = self.data.sum(axis = 1)
        ax3.plot(y, np.arange(len(y)))

        ax3.set_ylim(512 // self.fscrunch, 0)
        if self.save:
            name = self.fname.split('.dada')[0].split('/')[-1] + '_beam_' + self.beam + '.png'
            plt.savefig(name)
        else:
            plt.show()


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('-f', '--file_name', help = 'Name of the dada file.', type=str)
    a.add_argument('-n', '--num_samples', help = 'Number of time samples to plot.', type=int, default = 8192)
    a.add_argument('-ss', '--start_sample', help = 'Start sample- not required if the data is to be plotted from the start.', type=int, default = None)
    a.add_argument('-es', '--end_sample', help = 'End sample- not required if the data is to be plotted from the start.', type=int, default = None)
    a.add_argument('-dd', '--dedispersion_dm', help = 'Dedispersion DM.', type=float, default=None)
    a.add_argument('-ts', '--tscrunch', help = 'Factor by with time data should be tscrunched', type=int, default=None)
    a.add_argument('-fs', '--fscrunch', help = 'Factor by with frequency data should be fscrunched', type=int, default=None)
    a.add_argument('-b', '--beam', help = 'Beam in the file to be plotted', type=int, default=0)
    a.add_argument('-s', '--save', help = 'Save figure instead of plotting in the current directory', action='store_true', default=False)

    args = a.parse_args()
    Dada = ReadDada(args)
    Dada.get_data()

    if args.dedispersion_dm != None:
        Dada.dedisperse()
    if args.tscrunch != None:
        Dada.tdecimate()
    if args.fscrunch != None:
        Dada.fdecimate()

    Dada.Plot()

