# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:21:03 2015

@author: prlz7
"""

CAFFE_PATH='/home/prlz77/Code/caffe/python'
from loader import *
import sys
import lmdb
from PIL import Image
import StringIO
import pylab

class LMDB_ITERATOR(object):
    def __init__(self, path):
        self.path = path
        self.iterator = self.get_lmdb_iterator(path)
        for i, it in enumerate(self.iterator):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(self.iterator.value())
            label = datum.label
            try:
                img = np.array(bytearray(datum.data)).reshape(datum.channels, datum.height, datum.width).transpose((1,2,0))
                self.decoder = self.bytearray_decoder
            except:
                img = Image.open(StringIO.StringIO(datum.data))
                self.decoder = self.string_decoder
            break
        self.reset_iterator()
    def next_image(self):
        datum = caffe.proto.caffe_pb2.Datum()
        try:
            self.iterator.next()
            datum.ParseFromString(self.iterator.value())
        except StopIteration:
            return None
        label = datum.label
        return self.decoder(datum), label

    def reset_iterator(self):
        self.iterator = self.get_lmdb_iterator(self.path)
    def bytearray_decoder(self, datum):
        return np.array(bytearray(datum.data)).reshape(datum.channels, datum.height, datum.width).transpose((1,2,0))
    def string_decoder(self, datum):
        return Image.open(StringIO.StringIO(datum.data))
    def get_lmdb_iterator(self, path):
        self.env = lmdb.open(path)
        txn = env.begin()
        return txn.cursor()

def get_lmdb_iterator(path, stats=False):
    env = lmdb.open(path)
    txn = env.begin()
    if stats:
        return env.stat(), txn.cursor()
    else:
        return txn.cursor()
if __name__=='__main__':
    INPUT=sys.argv[1]
    
    env = lmdb.open(INPUT)
    txn = env.begin()
    cursor = txn.cursor()
    
    count = 0
    for i in cursor:
        count += 1
        
    print count, " images"
    
    cursor = txn.cursor()
    
    for i in cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(cursor.value())
        try:
            img = np.array(bytearray(datum.data)).reshape(datum.channels, datum.height, datum.width).transpose((1,2,0))
        except:
            img = Image.open(StringIO.StringIO(datum.data))
        print(np.array(img).max())
        print(np.array(img).min())
        pylab.imshow(img)
        pylab.title(str(datum.label))
        pylab.draw()
        print datum.label
        pylab.waitforbuttonpress()
        pylab.cla()
