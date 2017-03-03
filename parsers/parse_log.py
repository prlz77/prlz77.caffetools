# -*- coding: utf-8 -*-
# Author: prlz77 <pau.rodriguez at gmail.com>
# Group: ISELAB@CVC-UAB
# Date: 05/07/2016
""" Parses a Caffe Log """

def parse_caffe(path, output_fields=['accuracy']):
    """ Given a path

    :param path: ``string`` the log path.
    :param output_fields: ``list`` of ``string`` with the fields to 
    look for. Default is ['accuracy']
    """
    with open(path, 'r') as infile:
        lines = infile.readlines()
        test = False
        train_log = []
        test_log = []
        for l in lines:
            if 'Iteration' in l and  'Testing net' in l:
                test = True
                iteration = int(l.split('Iteration ')[1].split(',')[0])
                test_log.append({'iter':iteration})
            elif 'Iteration' in l:
                test = False
                iteration = int(l.split('Iteration ')[1].split(',')[0])
                train_log.append({'iter':iteration})
                if 'loss' in l:
                    loss = float(l.split('loss = ')[-1])
                    train_log[-1]['loss'] = loss
            elif test:
                found_field = False
                for field in output_fields:
                    if field in l:
                        accuracy = float(l.split('%s = '%field)[-1])
                        test_log[-1][field] = accuracy
                        found_field = True
                        break
                if not found_field and 'loss' in l:
                    loss = float(l.split('loss = ')[-1].split(' ')[0])
                    test_log[-1]['loss'] = loss

    return train_log, test_log  
