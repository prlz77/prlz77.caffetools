# -*- coding: utf-8 -*-
# Author: prlz77 <pau.rodriguez at gmail.com>
# Group: ISELAB@CVC-UAB
# Date: 07/03/2017
""" Plots a json parsed log """

import argparse
import numpy as np
import pylab
import os
import json
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=str, nargs="+", help="Files to plot. Filenames are used in legend")
    parser.add_argument("--legend", type=str, nargs="?", help="User defined legend. Note that #input files must equal the #input legend strings")
    parser.add_argument("--plot_fields", type=str, nargs="+", default=["accuracy"], help="Which curves to show")
    args = parser.parse_args()
    


    pylab.figure()
    pylab.hold(True)
    legend = []
    for i in args.filenames:
        if os.path.isfile(i):
            with open(i, 'r') as finput:
                data = json.load(finput)
        train = data['train']
        test = data['test']
        curves = { x:[] for x in args.plot_fields }
        test_iter = []
        for v in test:
            for field in args.plot_fields:
                if field in v:
                    curves[field].append(v[field])
                else:
                    curves[field].append(np.nan)
            test_iter.append(v['iter'])
        test_iter = np.array(test_iter, dtype=float)
        test_iter /= np.max(test_iter)
        for curve in curves:
            legend.append(i + " " + curve)
            pylab.plot(test_iter, np.array(curves[curve]))
    pylab.legend(legend).draggable()
    pylab.show()
