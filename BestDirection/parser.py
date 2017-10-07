import numpy as np
import os
def parse(index,database_path=''):
    def line_parser(f):
        s = f.readline()
        return list(map(eval,s.strip().split(' ')))

    relative = '../benchmark/db/'
    partindex = str(index // 100)
    fileindex = str(index)
    path = os.path.join(database_path,relative+partindex+'/m'+fileindex+'/m'+fileindex+'.off')

    f = open(path,'r')
    f.readline()
    V,F,_ = line_parser(f)
    vs = np.zeros(shape=(V,3))
    fs = np.zeros(shape=(F,3),dtype=int)
    for i in range(V):
        vs[i] = np.array(line_parser(f))

    for i in range(F):
        fs[i] = np.array(line_parser(f))[1:4]

    rng = np.random.RandomState(0)
    fc = np.ones((V, 4), dtype=np.float32)
    fc[:, 0] = np.linspace(1, 0, V)
    fc[:, 1] = rng.randn(V)
    fc[:, 2] = np.linspace(0, 1, V)

    return vs,fs,fc

def load(filename):
    res = list()
    if not os.path.isfile(filename):
        return res

    f = open(filename,'r')
    for l in f:
        res += [tuple(map(eval,l.strip().split(' ')))]

    return res
