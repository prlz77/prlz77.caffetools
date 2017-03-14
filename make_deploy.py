# -*- coding: utf-8 -*-
# Author: prlz77 <pau.rodriguez at gmail.com>
# Group: ISELAB@CVC-UAB
# Date: 17/11/2016
""" make_deploy
Creates a deploy.prototxt from a train_val.prototxt without third party dependencies
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import parsers.prototxt_parser as pparser
from warnings import warn

def make_deploy(train_val, input_size, last_layer=None):
    with open(train_val, 'r') as infile:
        trainval = infile.read()

    buffer = ""

    layer, trainval = pparser.get_next_layer(trainval)
    type = pparser.get_field(layer, "type")
    while pparser.get_field(layer, "type")[0].lower() == "data":
        layer, trainval = pparser.get_next_layer(trainval)

    buffer += ("""
    input: "data"
    input_shape
    {
        dim: 1  # batchsize
        dim: %d  # number of colour channels - rgb
        dim: %d  # height
        dim: %d  # width
    }
    """ % tuple(input_size))

    if last_layer is not None:
        while layer != "" and pparser.get_field(layer, "name")[0].lower() != last_layer:
            buffer += layer
            layer, trainval = pparser.get_next_layer(trainval)
        if layer != "":
            buffer += layer
        else:
            warn("Layer %s not found!" % last_layer)
    else:
        buffer += layer
        buffer += trainval

    return buffer

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Creates a deploy file from a train_val.proto")
    parser.add_argument("train_val", type=str, help="The train_val.prototxt")
    parser.add_argument("input_size", type=int, nargs=3, help="The input size: channels, width, height")
    parser.add_argument("--last_layer", "-l", type=str, default=None, help="Last layer name")
    parser.add_argument("--output", "-o", type=str, default="deploy.prototxt", help="The output file")
    args = parser.parse_args()

    with open(args.output, 'w') as deploy:
        deploy.write(make_deploy(args.train_val, args.input_size, args.last_layer))


