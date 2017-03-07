# -*- coding: utf-8 -*-
# Author: prlz77 <pau.rodriguez at gmail.com>
# Group: ISELAB@CVC-UAB
# Date: 03/03/2017
""" Parses a Caffe Log """

def parse_caffe(path, output_fields=['accuracy']):
    """ Given a path

    :param path: ``string`` the log path.
    :param output_fields: ``list`` of ``string`` with the fields to 
    look for. Default is ['accuracy']

    :return: train and test ordered lists.
    """
    with open(path, 'r') as infile:
        lines = infile.readlines()
        test = False
        train_log = []
        test_log = []
        for l in lines:
            if 'Iteration' in l and  'Testing net' in l:
                test = True
                iteration = int(l.split('Iteration ')[1].split(',')[0].split(' ')[0])
                test_log.append({'iter':iteration})
            elif 'Iteration' in l:
                test = False
                iteration = int(l.split('Iteration ')[1].split(',')[0].split(' ')[0])
                train_log.append({'iter':iteration})
                if 'loss' in l:
                    loss = float(l.split('loss = ')[-1].split(' ')[0])
                    train_log[-1]['loss'] = loss
            elif test:
                found_field = False
                for field in output_fields:
                    if field + " =" in l:
                        accuracy = float(l.split('%s = '%field)[-1].split(' ')[0])
                        test_log[-1][field] = accuracy
                        found_field = True
                        break
                if not found_field and 'loss =' in l:
                    loss = float(l.split('loss = ')[-1].split(' ')[0].split(' ')[0])
                    test_log[-1]['loss'] = loss

    return train_log, test_log  

if __name__ == '__main__':
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Extracts caffe log info")
    parser.add_argument("filename", type=str, help="log_path")
    parser.add_argument("--root_dir", type=str, default=None, help="If set \
            parse all logs with a given filename inside the subfolders of the \
            root dir")
    parser.add_argument("--output_fields", type=str, default=["accuracy"], 
            help="Which data to look for")
    args = parser.parse_args()

    if args.root_dir is None:
        train_log, test_log = parse_caffe(args.filename, args.output_fields)
        with open(args.filename + '.json', 'w') as output:
            json.dump({'train':train_log,'test':test_log}, output)
    else:
        dirlist = os.listdir(args.root_dir)
        for d in dirlist:
            path = os.path.join(args.root_dir, d, args.filename)
            if os.path.isfile(path):
                train_log, test_log = parse_caffe(path, args.output_fields)
                with open(d + '.json', 'w') as output:
                    json.dump({'train':train_log,'test':test_log}, output)
                
