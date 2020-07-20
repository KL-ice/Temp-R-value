from __future__ import division
from __future__ import print_function
import os
import sys
import glob
import time
import math
import mpmath
import random
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Process, Pool

import torch
import torch.nn as nn
import torch.nn.functional as F

from dLoader import *
from config import *
from method import *
from Entropy import *
import copy

def run_proc(E, n):

    E.cal_entropy_t()
    E.cal_entropy_st()

    f = open('./beijing/t_entropy_'+str(n)+'.txt', 'w')
    f.write(str(E.t_entropy))
    f.close()
    f = open('./beijing/t_pai_'+str(n)+'.txt', 'w')
    f.write(str(E.t_pai))
    f.close()

    f = open('./beijing/st_entropy_'+str(n)+'.txt', 'w')
    f.write(str(E.st_entropy))
    f.close()
    f = open('./beijing/st_pai_'+str(n)+'.txt', 'w')
    f.write(str(E.st_pai))
    f.close()

    return

if __name__ == '__main__':
    pools = 28

    roads = []
    with open('../data-' + ROAD_SET + '/' + ROAD_SET + '_roadSubset', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            roads += row
    print(roads)
    dataSet = networkRoadDataset(ROAD_SET)
    batch_len = dataSet.loadDataSet(roads, NEIGHBOUR_LEVEL, in_window = INPUT_WIDTH, delay_window = DELAY_WIDTH, out_window = OUTPUT_WIDTH)
    
    t_loader = dataSet.getBatch(roads)
    adj = dataSet.getAdj(roads)
    st_loader = dataSet.getNeighborBatch(roads, NUMPY(adj), level=4)

    t_loader = np.array_split(t_loader, pools)
    st_loader = np.array_split(st_loader, pools)

    for i in range(pools):
        print(len(t_loader[i]))
    print(len(t_loader))
    
    #创建对象
    Ents = []
    for i in range(pools):
        e = Entropy(t_loader[i], st_loader[i], i)
        Ents.append(e)
    
    #创建线程并启动线程
    p = Pool(pools)
    for i in range(pools):
        p.apply_async(run_proc, args=(Ents[i],i))

    #启动线程
    print('Waiting for all subprocesses done...')

    #等待终止
    p.close()
    p.join()
    print('All subprocesses done.')
