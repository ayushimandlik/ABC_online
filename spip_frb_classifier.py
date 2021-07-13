"""
# spip_frb_classifier_test.py

Script that reads in fan beam data and runs some test case classification
"""
import bifrost as bf
from blocks.dada_header_spip_A import dada_dict_to_bf_dict
from argparse import ArgumentParser
import sys
from adjbeam_classify import adjbeam_classify
from keras.models import load_model
from pprint import pprint
from datetime import datetime
import numpy as np
#from fdmt_block_blocks import FDMT
from fdmt_block_blocks_freq_in_kernal import FDMT
#from fdmt_block_blocks_v1 import FDMT
#from fdmt_block_blocks_getArgMax import FDMT
from NormaliseBlock import normalise
from glob import glob
import time
from keras.models import model_from_json


#class SaveStuffBlock(bf.pipeline.SinkBlock):
#    def __init__(self, iring, n_gulp_per_print=1, print_on_data=True, *args, **kwargs):
#        super(SaveStuffBlock, self).__init__(iring, *args, **kwargs)
#        self.n_iter = 0
#        self.n_gulp_per_print = n_gulp_per_print
#        self.print_on_data = print_on_data
#
#    def on_sequence(self, iseq):
#        print("[%s]" % datetime.now())
#        self.n_iter = 0
#
#    def on_data(self, ispan):
#        now = datetime.now()
#        if self.n_iter % self.n_gulp_per_print == 0 and self.print_on_data:
#            d = ispan.data
#            np.save('sft_data_for_fdmt_' + str(self.n_iter) , d)
#            print('Saving block: ' + str(ispan.data.shape) + str(ispan.data.dtype))
#        self.n_iter += 1

class PrintStuffBlock(bf.pipeline.SinkBlock):
    def __init__(self, iring, identifier, n_gulp_per_print=1, print_on_data=True, *args, **kwargs):
        super(PrintStuffBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0
        self.n_gulp_per_print = n_gulp_per_print
        self.print_on_data = print_on_data
        self.identifier = identifier

    def on_sequence(self, iseq):
        print("[%s]" % datetime.now())
        self.n_iter = 0

    def on_data(self, ispan):
        now = datetime.now()
        if self.n_iter % self.n_gulp_per_print == 0 and self.print_on_data:
            d = ispan.data
            print("[%s] %s %s %s" % (now, str(ispan.data.shape), str(ispan.data.dtype),
              self.identifier))
        self.n_iter += 1

if __name__ == "__main__":
    p = ArgumentParser(description='Multi-beam FRB classifier.')
    p.add_argument('-b', '--inputbuffer', default=0xFEFE, type=lambda x: int(x,0), help="PSRDADA buffer to connect to")
    p.add_argument('-f', '--filename', default=None, type=str, help="If set, will read from file instead of PSRDADA.")
    p.add_argument('-c', '--core', default=0, type=int, help="CPU core to bind input thread to, subsequent processes bind to subsequent cores")
    p.add_argument('--verbose',      default=False, action='store_true', help='Print out lots of stuff')

    args = p.parse_args()

    hdr_callback = dada_dict_to_bf_dict
#    model = load_model('/home/amandlik/ABC_direct_classifier/configa.hdf5')
#    model.summary()
    
    #json_file = open('/home/amandlik/ABC_scripts/model.json', 'r')
    #print("opened json file")
    #loaded_model_json = json_file.read()
    #json_file.close()
    #model1 = model_from_json(loaded_model_json)
    #model1.load_weights("/home/amandlik/ABC_scripts/weights.best.5D_FINAL_learning_rate_0.001_batch_150_epochs_140.hdf5")
    #model1.summary()

#    model1 = load_model("/home/amandlik/ABC_scripts/weights.best.5D_FINAL_learning_rate_0.001_batch_150_epochs_140.hdf5")
#    model1.summary()


    # Read in the data (either from a DADA buffer, or from disk)
    if args.filename is None:
        b_gpu = bf.blocks.psrdada.read_psrdada_buffer(args.inputbuffer, hdr_callback, 1, single=True, core=args.core, space='cuda')
    else:
        b_file = bf.blocks.read_dada_file(args.filename.split(','), hdr_callback, gulp_nframe=1, core=args.core)
        b_gpu  = bf.blocks.copy(b_file, space='cuda', core=args.core+1, gpu=1)
        i = 0

    with bf.block_scope(fuse=True, gpu=1):
        # Normalised data block
        start = datetime.now()
        n_gpu = normalise(b_gpu)
        a_gpu  = bf.blocks.transpose(n_gpu, ['time','freq','fine_time', 'beam'])
        a_gpu  = bf.views.split_axis(a_gpu, 'beam', 3, label='fine_beam')
        #a_gpu  = bf.views.split_axis(a_gpu, 'beam', 6, label='fine_beam') # for batch_size = 4
        a_gpu = bf.views.rename_axis(a_gpu, 'beam', 'batch')
        a_gpu  = bf.blocks.transpose(a_gpu, ['time', 'batch', 'freq', 'fine_time', 'fine_beam'])
        a_gpu = bf.blocks.reduce(a_gpu, 'freq', 2, op='mean')
        adjbeam_classify(a_gpu)
        PrintStuffBlock(a_gpu, 'a')
        #print(datetime.now() - start)
        b_gpu_sft =  bf.blocks.transpose(n_gpu, ['time','beam','freq', 'fine_time'])
#        SaveStuffBlock(b_gpu_sft)
        o_gpu = FDMT(b_gpu_sft)
        PrintStuffBlock(o_gpu, 'abc')

 #       print(datetime.now() - start)

 #   print(bf.get_default_pipeline().dot_graph())
    bf.get_default_pipeline().shutdown_on_signals()
    bf.get_default_pipeline().run()

                                  


