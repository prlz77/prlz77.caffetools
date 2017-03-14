# -*- coding: utf-8 -*-
# Author: prlz77 <pau.rodriguez at gmail.com>
# Group: ISELAB@CVC-UAB
# Date: 17/11/2016
""" gen_deploy
Creates a deploy.prototxt from a train_val.prototxt without third party dependencies
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import prototxt_parser as pparser
from warnings import warn

parser = argparse.ArgumentParser(description="Creates a deploy file from a train_val.proto")
parser.add_argument("train_val", type=str, help="The train_val.prototxt")
parser.add_argument("input_size", type=int, nargs=3, help="The input size: channels, width, height")
parser.add_argument("--last_layer", "-l", type=str, default=None, help="Last layer name")
parser.add_argument("--output", "-o", type=str, default="deploy.prototxt", help="The output file")
args = parser.parse_args()

with open(args.train_val, 'r') as infile:
    trainval = infile.read()

deploy = open(args.output, 'w')

layer, trainval = pparser.get_next_layer(trainval)
type = pparser.get_field(layer, "type")
while pparser.get_field(layer, "type")[0].lower() == "data":
    layer, trainval = pparser.get_next_layer(trainval)

deploy.write("""
input: "data"
input_shape
{
    dim: 1  # batchsize
    dim: %d  # number of colour channels - rgb
    dim: %d  # height
    dim: %d  # width
}
""" % tuple(args.input_size))

if args.last_layer is not None:
    while layer != "" and pparser.get_field(layer, "name")[0].lower() != args.last_layer:
        deploy.write(layer)
        layer, trainval = pparser.get_next_layer(trainval)
    if layer != "":
        deploy.write(layer)
    else:
        warn("Layer %s not found!" % args.last_layer)
else:
    deploy.write(layer)
    deploy.write(trainval)

deploy.close()
