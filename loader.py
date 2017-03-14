import sys
import os
sys.path.insert(0, os.path.realpath(__file__))
import config
from config import CAFFE_PATH
sys.path.insert(0, os.path.join(CAFFE_PATH, 'python'))
sys.path.insert(1, os.path.join(CAFFE_PATH, 'tools/extra'))
sys.path.insert(1, '/home/prlz77/libs/NPEET')
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
from utils import *
from parsers import *
from monitors import *
