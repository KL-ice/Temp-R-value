from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import time
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
    # batch_len = 10
    train_break = int(0.6 * batch_len)
    validate_break = train_break + int(0.15 * batch_len)
    idx_train = range(train_break)
    idx_val = range(train_break, validate_break)
    idx_test = range(validate_break, batch_len)

    loader = dataSet.getBatch(roads)
    print(roads)

    adj = dataSet.getAdj(roads)
    neighborLoader = dataSet.getNeighborBatch(roads, NUMPY(adj), level=3)



    '''
    E = entropys()
    S_train = E.get_pai_t(loader, idx_train)
    S_val = E.get_pai_t(loader, idx_val)
    S_test = E.get_pai_t(loader, idx_test)

    print((S_test[0]+S_train[0]+S_val[0])/3)
    print((S_test[1]+S_train[1]+S_val[1])/3)
    print((S_test[2]+S_train[2]+S_val[2])/3)
    print((S_test[3]+S_train[3]+S_val[3])/3)
    print((S_test[4]+S_train[4]+S_val[4])/3)
    print((S_test[5]+S_train[5]+S_val[5])/3)

    f = open('Srand.txt', 'w')
    f.write(str((S_test[0]+S_train[0]+S_val[0])/3))
    f.close()

    f = open('Sunc.txt', 'w')
    f.write(str((S_test[1]+S_train[1]+S_val[1])/3))
    f.close()

    f = open('Sact.txt', 'w')
    f.write(str((S_test[2]+S_train[2]+S_val[2])/3))
    f.close()

    f = open('pairand.txt', 'w')
    f.write(str((S_test[3]+S_train[3]+S_val[3])/3))
    f.close()

    f = open('paiunc.txt', 'w')
    f.write(str((S_test[4]+S_train[4]+S_val[4])/3))
    f.close()

    f = open('paiact.txt', 'w')
    f.write(str((S_test[5]+S_train[5]+S_val[5])/3))
    f.close()
    '''
    '''
    计算 st
    E = entropys()
    S_sttrain = E.get_pai_st(neighborLoader, idx_train)
    S_stval = E.get_pai_st(neighborLoader, idx_val)
    S_sttest = E.get_pai_st(neighborLoader, idx_test)

    f = open('Sst.txt', 'w')
    f.write(str((S_sttest[0]+S_sttrain[0]+S_stval[0])/3))
    f.close()

    f = open('paist.txt', 'w')
    f.write(str((S_sttest[1]+S_sttrain[1]+S_stval[1])/3))
    f.close()
    '''
    E = entropys()
    S_sttrain = E.get_pai_s(neighborLoader, idx_train)
    S_stval = E.get_pai_s(neighborLoader, idx_val)
    S_sttest = E.get_pai_s(neighborLoader, idx_test)

    f = open('Ss.txt', 'w')
    f.write(str((S_sttest[0]+S_sttrain[0]+S_stval[0])/3))
    f.close()

    f = open('pais.txt', 'w')
    f.write(str((S_sttest[1]+S_sttrain[1]+S_stval[1])/3))
    f.close()
