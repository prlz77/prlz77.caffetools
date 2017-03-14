# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:21:03 2015

@author: prlz7
"""

from loader import caffe_pb2
import sys
import lmdb
import pylab
from warnings import warn
import cv2
import numpy as np

class LMDB_ITERATOR(object):
    def __init__(self, path):
        self.path = path
        self.iterator = self.get_lmdb_iterator(path)
        for i, it in enumerate(self.iterator):
            datum = caffe_pb2.Datum()
            datum.ParseFromString(self.iterator.value())
            label = datum.label
            try:
                img = np.array(bytearray(datum.data)).reshape(datum.channels, datum.height, datum.width).transpose((1,2,0))
                self.decoder = self.bytearray_decoder
            except:
                arr = np.fromstring(datum.data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                #img = Image.open(StringIO.StringIO(datum.data))
                self.decoder = self.string_decoder
            self.imsize = img.shape
            self.n = self.env.stat()['entries']
            break
        self.reset_iterator()

    def next_image(self):
        datum = caffe_pb2.Datum()
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
        txn = self.env.begin()
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
        datum = caffe_pb2.Datum()
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
