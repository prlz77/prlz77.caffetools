# -*- coding: utf-8 -*-
# Author: prlz77 <pau.rodriguez at gmail.com>
# Group: ISELAB@CVC-UAB
# Date: 07/03/2017

import os
from datetime import datetime
import argparse
import psutil
import sys
import time
import signal
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from parsers.parse_log import parse_caffe

def field2array(log, field):
    history = [0]
    for v in log:
        if field in v:
            history.append(v[field])
    history = np.array(history)
    return history

def is_plateau(history, plateau_size):
    return history.shape[0] - np.argmax(history) > plateau_size


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int, help="process pid")
    parser.add_argument("log_path", type=str, help="caffe.INFO path")
    parser.add_argument("plateau_size", type=int, help="Number of times without an improvement.")
    parser.add_argument("--field", '-f', type=str, default="accuracy", help="Field to monitor")
    parser.add_argument("--time", '-t', type=int, default=60, help="Refresh time in seconds")
    args = parser.parse_args()

    assert(psutil.pid_exists(args.pid))

    _, test_log = parse_caffe(args.log_path, [args.field])
    history = field2array(test_log, args.field)

    while psutil.pid_exists(args.pid) and not is_plateau(history, args.plateau_size):
        print "[%s] PID: %d, Log: %s" %(str(datetime.now()), args.pid, args.log_path)
        print "Current iter: %d, Current accuracy: %f" %(history.shape[0], history[-1])
        print "Best iter %d, Best accuracy: %f, not in plateau." %(np.argmax(history), np.max(history))
        time.sleep(args.time)
        _, test_log = parse_caffe(args.log_path, [args.field])
        history = field2array(test_log, args.field)
    if psutil.pid_exists(args.pid):
        print "Plateau reached"
        print "Best iter %d, Best accuracy: %f." %(np.argmax(history), np.max(history))
        psutil.Process(args.pid).send_signal(signal.SIGINT)
        print "Waiting process to exit"
        while psutil.pid_exists(args.pid):
            time.sleep(args.time)
        print "Done."
    else:
        print "Process stopped before plateau."
        


