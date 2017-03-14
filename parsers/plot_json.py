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
    parser.add_argument("filenames", type=str, nargs="+", 
            help="Files to plot. Filenames are used in legend")
    parser.add_argument("--legend", type=str, nargs="?", 
            help="User defined legend. Note that #input files must equal the #input legend strings")
    parser.add_argument("--plot_fields", type=str, nargs="+", 
            default=["accuracy"], help="Which curves to show")
    parser.add_argument("--filter_style", type=str, nargs="+", default=[], 
            help="ex: --filter_style baseline dashed; will use dashed style for all the inputs containing the text *baseline*")
    parser.add_argument("--ignore_pattern", type=str, nargs="+", default=[],
            help="Will ignore all files containing the specified substring")
    parser.add_argument("--max_curves", type=int, default=25)
    parser.add_argument("--no_sort", action="store_true")
    args = parser.parse_args()
    
    if len(args.filter_style) % 2 != 0:
        raise ValueError("Style filters must be multiple of two: pattern style")
    

    pylab.figure()
    pylab.hold(True)
    plot_data = []
    y_data = []

    for i in args.filenames:
        skip = False
        for pattern in args.ignore_pattern:
            if pattern in i:
                skip = True
                continue
        if skip:
            continue
        linestyle = '-'
        for p_idx in xrange(0, len(args.filter_style), 2):
            if args.filter_style[p_idx] in i:
                linestyle=args.filter_style[p_idx + 1]
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
        if len(test_iter) == 0:
            continue
        test_iter = np.array(test_iter, dtype=float)
        test_iter /= np.max(test_iter)
        for j, curve in enumerate(curves):
            if len(curves[curve]) > 1 and len(test_iter) > 1:
                plot_data.append({"x":test_iter, "y":curves[curve], "legend":"%s %s" %(i, curve), "linestyle":linestyle})
                y_data.append(curves[curve][-1])
            #pylab.plot(test_iter, np.array(curves[curve]), linestyle=linestyle, c=cmap[len(legend) - 1])
            #legend.append(i + " " + curve)

    srt = np.argsort(y_data)[::-1]
    cmap = sns.color_palette("Set1", n_colors=len(srt), desat=.5)
    legend = []

    for i in srt:
        print cmap[i]
        pylab.plot(plot_data[i]['x'], plot_data[i]['y'], linestyle=plot_data[i]['linestyle'], c=cmap[i])
        legend.append(plot_data[i]['legend'])
    pylab.legend(legend).draggable()
    pylab.show()
