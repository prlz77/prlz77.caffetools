# -*- coding: utf-8 -*-
# Author: prlz77 <pau.rodriguez at gmail.com>
# Group: ISELAB@CVC-UAB
# Date: 17/11/2016

def read_eol(text, ind):
    buff = text[ind]
    counter = 0
    while buff[-1] != '\n':
        counter += 1
        buff += text[ind+counter]

def match_symbol(text, sym):
    inside = False
    end = False
    buffer = ""
    counter = 0
    while not end:
        if not inside and text[counter] == sym:
            inside = True
        elif inside and text[counter] == sym:
            end = True
        elif inside and text[counter] != sym:
            buffer += text[counter]
        counter += 1
    return buffer

def match_symbol(text,symbol_start='{', symbol_end='}'):
    end = False
    stack = 0
    counter = 0
    if symbol_start in text and symbol_end in text:
        start = text.find(symbol_start)
        counter = start - 1
        while not end:
            counter += 1
            if stack == 0 and text[counter] == symbol_start:
                stack += 1
            elif stack > 0 and text[counter] == symbol_start:
                stack += 1
            elif stack > 1 and text[counter] == symbol_end:
                stack -= 1
            elif stack == 1 and text[counter] == symbol_end:
                end = True
    return start, counter

def get_next_layer(text):
    buffer = ""
    if "layer" in text:
        lpos = text.index('layer')
        buffer = text[lpos:]
        start, end = match_symbol(buffer)
        end += 1
    return buffer[:end], buffer[end:]

def get_field(text, field):
    lines = text.split('\n')
    ret = []
    for l in lines:
        if field.lower() in l.lower():
            l2 = l.replace("'",'').replace('"','').replace(" ", "")
            ret.append(l2.split(':')[-1])
    return ret

def set_field(text, field, values):
    lines = text.split('\n')
    for i,l in enumerate(lines):
        if field.lower() in l.lower():
            lines[i] = "%s : %s" %(field, str(values[0]))
            values = values[1:]
        if len(values == 0):
            break
    return '\n'.join(lines)


class ProtoParser():
    def __init__(self, text):
        self.text = text
        name_ind = self.text.find('name:')
        self.name = match_symbol(self.text[name_ind:])
        self.layers = []


