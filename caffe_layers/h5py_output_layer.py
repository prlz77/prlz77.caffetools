import sys
print sys.path
sys.path.insert(0, '/home/prodriguez/caffe/python')
import caffe
import h5py
import os
#sys.path.insert(0, os.path.dirname(__file__))
import json
import numpy as np
class H5PYOutput(caffe.Layer):
    f = None
    def setup(self, bottom, top):
        self.params = json.loads(self.param_str)
        if H5PYOutput.f is None:
            H5PYOutput.f = h5py.File(self.params['output_file'], 'w')
            for i in xrange(len(bottom)):
                print bottom[i].data[...].shape
                mshape = list(bottom[i].data[...].shape)
                if len(mshape) == 1:
                    mshape = [None]
                else:
                    mshape[0] = None
                shape = [j for j in mshape]
                shape[0] = 0
                H5PYOutput.f.create_dataset(self.params['bottoms'][i], tuple(shape), dtype='float32', maxshape=mshape)
	self.bs = bottom[0].data[...].shape[0]
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for i in xrange(len(bottom)):
            layer = self.params['bottoms'][i]
            size = list(H5PYOutput.f[layer].shape)
            size[0] = size[0] + self.bs
            print size
            H5PYOutput.f[layer].resize(size)
            H5PYOutput.f[layer][-self.bs:,...] = bottom[i].data[...]
    def backward(self, top, propagate_down, bottom):
        pass
