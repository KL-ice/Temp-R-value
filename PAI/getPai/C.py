from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import time
import multiprocessing
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dLoader import *
from config import *
from method import *
from readData import entropys


if __name__ == "__main__":
    roads = []
    with open('../data-' + ROAD_SET + '/road_net.json') as f:
        road_net = json.load(f)
    with open('../data-' + ROAD_SET + '/' + ROAD_SET + '_roadSubset', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            roads += row
    dataSet = networkRoadDataset(ROAD_SET)
    batch_len = dataSet.loadDataSet(roads, NEIGHBOUR_LEVEL, in_window = INPUT_WIDTH, delay_window = DELAY_WIDTH, out_window = OUTPUT_WIDTH)

    idx1 = range(int(0.18*batch_len))
    idx2 = range(int(0.18*batch_len), int(0.35*batch_len))
    idx3 = range(int(0.35*batch_len), int(0.50*batch_len))
    idx4 = range(int(0.50*batch_len), int(0.68*batch_len))
    idx5 = range(int(0.68*batch_len), int(0.85*batch_len))
    idx6 = range(int(0.85*batch_len), 1*batch_len)

    adj = dataSet.getAdj(roads)
    neighborLoader = dataSet.getNeighborBatch(roads, NUMPY(adj), level=3)
   
    # 创建线程
    E1 = entropys(neighborLoader, idx1, 1)
    E2 = entropys(neighborLoader, idx2, 2)
    E3 = entropys(neighborLoader, idx3, 3)
    E4 = entropys(neighborLoader, idx4, 4)
    E5 = entropys(neighborLoader, idx5, 5)
    E6 = entropys(neighborLoader, idx6, 6)

    #开启线程
    print('starting process')
    E1.start()
    E2.start()
    E3.start()
    E4.start()
    E5.start()
    E6.start()

    print('already start')
    process = []
    # 添加线程到线程列表
    process.append(E1)
    process.append(E2)
    process.append(E3)
    process.append(E4)
    process.append(E5)
    process.append(E6)

    for t in process:
        t.join()

    print('done')