# -*- coding:utf-8 -*-
def read_int_lines(fname):
    lines = []
    fd = open(fname)
    for line in fd:
        lines.append(int(line.strip()))
    fd.close()    
    return lines

def shuffle_multiple_lst(*lst):
    import random
    if len(lst) > 0:
        index_lst = [i for i in range(len(lst[0]))]    
        random.shuffle(index_lst)
        new_lst = [index_lst]
        for l in lst:
            new_lst.append([l[i] for i in index_lst])
        return new_lst        
    else:
        None  

def read_str_to_id_map(fname, sep):
    import codecs
    k2v = {}
    v2k = {}
    fd = codecs.open(fname, 'r', 'utf-8')
    for line in fd:
        line = line.strip().split(sep)
        str_key = line[0]
        id_value = int(line[1])
        k2v[str_key] = id_value
        v2k[id_value] = str_key
    fd.close()
    return k2v, v2k

if __name__ == '__main__':
    pass