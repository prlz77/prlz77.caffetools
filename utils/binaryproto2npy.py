# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:30:36 2015

@author: prlz7
"""
from loader import *

def proto2npy(path):
    with open(path, 'r') as infile:
        blob = caffe_pb2.BlobProto().FromString(infile.read())

    return caffe.io.blobproto_to_array(blob)



if __name__ == '__main__':
    path = sys.argv[1]
    blob_array = proto2npy(path)
    if len(sys.argv) < 3:
        sys.argv.append(sys.argv[1] + '.npy')
    np.save(sys.argv[2], blob_array[0])


