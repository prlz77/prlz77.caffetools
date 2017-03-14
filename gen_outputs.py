# -*- coding: utf-8 -*-
""" Generates the outputs of an arbitrary CNN layer. """

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import argparse

# CMD Options
parser = argparse.ArgumentParser(description="""Generates the outputs of an arbitrary CNN layer
accepts either a LMDB dataset or a listfile of images.""")
parser.add_argument('model', type=str, help='The model deploy file.')
parser.add_argument('weights', type=str, help='The model weights file.')
parser.add_argument('layer', type=str, nargs='+', help='The target layer(s).')
parser.add_argument('--output', type=str, help='The output file.', default='output.h5')
parser.add_argument('--flist', nargs=2, type=str, help='The base folder and the file list of the images.', default=None)
parser.add_argument('--label_names', nargs='+', type=str, default=['labels'], help='specific label names, accepts more than one label')
parser.add_argument('--dataset', type=str, help='The lmdb dataset.', default=None)
parser.add_argument('--mean_tensor', type=float, nargs=3, default=None, help='Pixel mean (3 bgr values)')
parser.add_argument('--mean_file', type=str, default=None, help='Per-pixel mean in bgr')
parser.add_argument('--raw_scale', type=float, default=None, help='Scale value before mean subtraction.')
parser.add_argument('--input_scale', type=float, default=None, help='Scale value after mean subtraction.')
parser.add_argument('--swap', action='store_true', help='BGR -> RGB. Make sure lmdb raw images are bgr')
parser.add_argument('--cpuonly', action='store_true', help='CPU-Only flag.')
parser.add_argument('--standarize', action='store_true', help="whether to standarize the outputs")
parser.add_argument('--standarize_with', type=str, default='', help='get mean and std from another .h5 (recommended for validation)')
parser.add_argument('--verbose', action='store_true', help='show image paths while being processed')
parser.add_argument('--batch_size', '-b', type=int, default=1, help="batch size")
parser.add_argument('--make_deploy', '-d', nargs=3, default=[], type=int, help="given the input size (c h w), it converts trainval into deploy")
parser.add_argument('--flatten', action="store_true", help="whether to save flattened features")
parser.add_argument('--center_crop', type=int, default=None, help="Center crop size")
parser.add_argument('--test', action="store_true")

args = parser.parse_args()

#TODO
if args.standarize and len(args.label_names > 1):
    raise NotImplementedError("This code does not support yet standarizing multiple labels")


# Move the rest of imports to avoid conflicts with argparse
import sys
import os
from loader import caffe
from PIL import Image
import h5py
import numpy as np
    
if len(args.make_deploy) > 0:
    from make_deploy import make_deploy
    model_path = os.path.join(os.path.dirname(args.model), "deploy.prototxt")
    with open(model_path, 'w') as deploy:
        deploy.write(make_deploy(args.model, args.make_deploy, args.layer[-1]))
    args.model = model_path


# CPU ONLY
if not args.cpuonly:
    caffe.set_mode_gpu()

# Read Deploy + Weights file
net = caffe.Net(args.model, args.weights,
                caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# images are assumed to be in format hxwxc
transformer.set_transpose('data', (2,0,1))

# get mean
if args.mean_tensor is not None:
    transformer.set_mean('data', np.array(args.mean_tensor))#np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
elif args.mean_file != None:
    if 'npy' in args.mean_file:
        mean = np.load(args.mean_file)
    else:
        from loader import binaryproto2npy as p2n
        mean = p2n.proto2npy(args.mean_file)[0]
    try:
        transformer.set_mean('data', mean)
    except:
        mean = mean.mean(1).mean(1)
        transformer.set_mean('data', mean)
if args.raw_scale:
    transformer.set_raw_scale('data', args.raw_scale)  # the reference model operates on images in [0,255] range instead of [0,1]
if args.input_scale:
    transformer.set_input_scale('data', args.input_scale)  # the reference model operates on images in [0,255] range instead of [0,1]
if args.swap:
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(args.batch_size, net.blobs['data'].data.shape[1], net.blobs['data'].data.shape[2], net.blobs['data'].data.shape[3])
if args.verbose:
    print "input shape: ", str([i for i in net.blobs['data'].shape])


if args.flist is not None:
    from cv2 import imread
    class FileListReader():
        def __init__(self, root_path, list_path, label_names=['labels']):
            with open(list_path, 'r') as infile:
                self.flist = infile.readlines()
            self.root_path = root_path
            self.imsize = cv2.imread(os.path.join(self.root_path, flist[0]))
            self.n = len(flist)
            self.index = 0
            self.label_names = label_names
        def get_next_image(self):
            labels = {}
            spline = self.flist[self.index].replace('\n', '').split(" ")
            for idx, label_name in enumerate(self.label_names):
                self.labels[label_name] = float(spline[1 + idx])
            img = imread(os.path.join(self.root_path, spline[0]).replace('\\', '/'))
            self.index += 1
            return img, labels
        def reset(self):
            self.index = 0
    reader = FileListReader(args.flist[0], args.flist[1], args.label_names)
elif args.dataset is not None:
    from loader import lmdb_visualizer as lmdbv
    class LMDBReader(lmdbv.LMDB_ITERATOR):
        def __init__(self, path, label_names = ["labels"]):
            super(LMDBReader, self).__init__(path)    
            self.label_names = label_names 
        def get_next_image(self):
            img, label = super(LMDBReader, self).next_image()
            return img, {"labels": label }
        def reset(self):
            super(LMDBReader, self).reset_iterator()
    reader = LMDBReader(args.dataset)
else:
    raise Exception('need a dataset')



labels = {}
for label_name in args.label_names:
    labels[label_name] = []
outputs = []
if args.center_crop is not None:
    pad_h = reader.imsize[0] - args.center_crop
    pad_w = reader.imsize[1] - args.center_crop
    sh = pad_h // 2
    eh = sh + args.center_crop
    sw = pad_w // 2
    ew = sw + args.center_crop
else:
    sh = 0
    eh = reader.imsize[0]
    sw = 0
    ew = reader.imsize[1]

for layer in args.layer:
    outputs.append(h5py.File(args.output + '_' + layer.replace('/','_') + '.h5', 'w'))
    dim = list(net.blobs[layer].data.shape[1:])
    if args.flatten:
        dim = [np.array(dim).prod()]
    elif len(dim) < 3:
        dim = [np.array(dim).prod(), 1, 1]
    outputs[-1].create_dataset('outputs', tuple([reader.n] + dim), dtype='float32')
    for label_name in args.label_names:
        outputs[-1].create_dataset(label_name, (reader.n,), dtype='float')

for i in xrange(reader.n):
    img, label_dic = reader.get_next_image()
    for idx, label_name in enumerate(args.label_names):
        labels[label_name].append(label_dic[label_name])
    net.blobs['data'].data[((1+i) % args.batch_size)-1,...] = transformer.preprocess('data', img[sh:eh,sw:ew,:])
    if (i+1) % args.batch_size == 0 or i + 1 == reader.n:
        out = net.forward()
        bs = net.blobs['data'].data.shape[0]
        for index, layer in enumerate(args.layer):
            outputs[index]['outputs'][(i-bs+1):i+1,...] = net.blobs[layer].data[...].reshape([-1] + dim)
            for label_name in args.label_names:
                outputs[index][label_name][(i-bs+1):i+1] = labels[label_name][-bs:]
        next_batch_length = reader.n - (i+1)
        if bs == args.batch_size and next_batch_length > 0 and next_batch_length < args.batch_size:             
            net.blobs['data'].reshape(next_batch_length, net.blobs['data'].data.shape[1], net.blobs['data'].data.shape[2], net.blobs['data'].data.shape[3])
    if i % 1000 == 0:
        print "Processing image ", i

for index, layer in enumerate(args.layer):
    if os.path.isfile(args.standarize_with):
        train = h5py.File(args.standarize_with, 'r')
        mean = train['mean'][...]
        std = train['std'][...]
        label_mean = train['label_mean'][...]
        label_std = train['label_std'][...]
    elif args.standarize:
        label_mean = outputs[index]['labels'][...].mean()
        label_std = outputs[index]['labels'][...].std()
        mean = outputs[index]['outputs'][...].mean()
        std = outputs[index]['outputs'][...].std()
    else:
        mean = 0.0
        std = 1.0

if args.standarize or os.path.isfile(args.standarize_with):
    outputs[index]['labels'][...] -= label_mean
    outputs[index]['labels'][...] /= label_std
    outputs[index]['label_mean'] = label_mean
    outputs[index]['label_std'] = label_std
    outputs[index]['outputs'][...] -= mean
    outputs[index]['outputs'][...] /= std
    outputs[index]['mean'] = mean
    outputs[index]['std'] = std

if args.test:
    print (outputs[-1]['outputs'][...].argmax(axis=1).flatten() == outputs[-1]['labels'][...].flatten()).mean()

for index, layer in enumerate(args.layer):
    outputs[index].close()
