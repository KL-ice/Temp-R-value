from __future__ import division
from __future__ import print_function
import math
import pandas as pd
import numpy as np
import mpmath
import torch

import os
import sys
import glob
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing

from dLoader import *
from config import *
from method import *


class entropys():

    def __init__(self, l, interval, number):
        self.l = l
        self.interval = interval
        self.number = number
        self.s_st = []
        self.paist = []
        self.s_s = []
        self.pai_s = []
        self.s_t = []
        self.pai_t = []

    def count_data(self, name, Dir):
        F = open(Dir+name, 'r')
        Datas = F.readlines()
        Datas = np.array(Datas, dtype=np.string_)
        Datas = Datas.astype(float).astype(np.int32)
        result = pd.value_counts(Datas)
        return Datas, len(result)

    def randEntropy(self, l):
        '''
        get the rand entropy
        input: l: road * in_window
        '''
        randE = []
        N = []
        for i in l:
            i = np.around(i.numpy(), decimals=2)
            count = pd.value_counts(i)
            n = len(count)
            S_rand = math.log(n, 2)
            randE.append(S_rand)
            N.append(n)
        return randE, N

    def uncEntropy(self, l):
        '''
        get the unc entropy
        input: l: road * in_window
        '''
        uncE = []
        for i in l:
            i = np.around(i.numpy(), decimals=2)
            count = pd.value_counts(i).as_matrix()
            n = np.sum(count)
            count = count/n * np.log2(count/n)
            S_unc = -np.sum(count)
            uncE.append(S_unc)
        return uncE

    def contains(self, small, big):
        '''
        Determine if the small is in the big
        '''
        for i in range(len(big)-len(small)+1):
            if big[i:i+len(small)] == small:
                return True
        return False

    def actualEntropy(self, l):
        '''
        get the actual entropy
        input: l: road * in_window
        '''
        actE = []
        for k in l:
            '''保留n位小数'''
            k = np.around(k.numpy(), decimals=2)
            n = len(k)
            sequence = [k[0]]
            sum_gamma = 0

            for i in range(1, n):
                for j in range(i+1, n+1):
                    s = list(k[i:j])
                    if self.contains(s, sequence) != True:
                        sum_gamma += len(s)
                        sequence.append(k[i])
                        break
            ae = 1 / (sum_gamma / n) * math.log(n, 2)
            actE.append(ae)
        return actE

    def stContains(self, small, big, h):
        '''
        Determine if the small is in the big around time
        '''
        h_small, w_small = small.shape

        for i in range(h):
            if i+h_small > h:
                break
            w = len(big[i])
            for j in range(w):
                if j+w_small > w:
                    break
                big_slice = []
                for inx in range(i, i+h_small):
                    if j+w_small > len(big[i]):
                        break
                    big_slice.append(big[inx][j:j+w_small])

                if big_slice == small.tolist():
                    return True
        return False

    def oneRoadStE(self, l):
        '''
        get the st entropy
        input: l: neighbor * in_window
        '''

        neighbor, time = l.shape
        n = max(neighbor, time)
        sum_gamma = 0
        sequence = []
        for i in range(n):
            sequence.append([])

        for i in range(neighbor):
            for j in range(time):
                for k in range(1, n):
                    if i+k >= neighbor and j+k >= time:
                        break
                    h = min(i+k, neighbor)
                    w = min(j+k, time)
                    s = l[i:h, j:w]
                    if not self.stContains(s, sequence, i):
                        sum_gamma += k
                        break
                sequence[i].append(l[i, j])

        if sum_gamma == 0 or n == 0:
            ae = 0
        else:
            ae = 1 / (sum_gamma / n) * math.log(n, 2)
        ni = np.unique(l).shape[0]
        return ae, ni

    def stEntropy(self, l):
        '''
        get the st entropy
        input: l: road * neighbor * in_window
        '''
        actE = []
        N = []
        for k in l:
            '''保留n位小数, k: neighbor * in_window'''
            k = np.around(k.numpy(), decimals=2)
            e, n = self.oneRoadStE(k)
            actE.append(e)
            N.append(n)
        return actE, N

    def sEntropy(self, l):
        '''
        get the s entropy
        input: l: road * neighbor * time
        '''
        S_s = []
        pai_s = []
        for i in l:    # i -> one road
            S = self.actualEntropy(i)   #96*1
            _, N = self.randEntropy(i)  #96*1
            pai = self.getPredictability(N, S)  #96*1
            S = np.mean(S)
            pai = np.mean(pai)
            S_s.append(S)
            pai_s.append(pai)
        return S_s, pai_s
            
    def getPredictability(self, N, S):
        '''
        slove the equation to get the pai
        '''
        pai = []
        for i, n in zip(S, N):
            if n > 1:
                def f(x): return (((1-x)/(n-1)) ** (1-x)) * x**x - 2**(-i)
                root = mpmath.findroot(f, 1)
                pai.append(float(root.real))
            else:
                pai.append(1)
        return pai

    def get_pai_t(self, loader, interval):
        S_rand = []
        S_unc = []
        S_act = []
        pai_rand = []
        pai_unc = []
        pai_act = []
        for i in interval:
            print(i)
            (x, y, y_max, y_mean) = next(loader)
            randS, N = self.randEntropy(x)
            S_rand.append(randS)
            S_unc.append(self.uncEntropy(x))
            S_act.append(self.actualEntropy(x))
            print(N)
            pai_rand.append(self.getPredictability(N, S_rand[-1]))
            pai_unc.append(self.getPredictability(N, S_unc[-1]))
            pai_act.append(self.getPredictability(N, S_act[-1]))
        S_rand = np.mean(np.array(S_rand), axis=0)
        S_unc = np.mean(np.array(S_unc), axis=0)
        S_act = np.mean(np.array(S_act), axis=0)
        pai_rand = np.mean(np.array(pai_rand), axis=0)
        pai_unc = np.mean(np.array(pai_unc), axis=0)
        pai_act = np.mean(np.array(pai_act), axis=0)

        return S_rand, S_unc, S_act, pai_rand, pai_unc, pai_act

    def get_pai_st(self, loader, interval):
        '''
        get the st entropy and pai
        '''
        S_st = []
        pai_st = []
        for i in interval:
            print(i)
            x = loader[i]
            sst, N = self.stEntropy(x)
            S_st.append(sst)

            print(N)
            pai_st.append(self.getPredictability(N, S_st[-1]))

        S_st = np.mean(np.array(S_st), axis=0)
        pai_st = np.mean(np.array(pai_st), axis=0)

        #return S_st, pai_st
        self.s_st = S_st
        self.paist = pai_st

    def get_pai_s(self, loader, interval):
        '''
        get the s entropy and pai
        roadd: road * neighbor * time
        '''
        S_s = []
        pai_s = []
        for i in interval:
            print(i)
            (x, y, y_max, y_mean) = next(loader)
            ss, p = self.sEntropy(x)
            S_s.append(ss)
            pai_s.append(p)

        S_s = np.mean(np.array(S_s), axis=0)
        pai_s = np.mean(np.array(pai_s), axis=0)

        return S_s, pai_s

    def run(self):
        self.get_pai_st(self.l, self.interval)
        f = open('./st/sst3_'+str(self.number)+'.txt', 'w')
        f.write(str(self.s_st))
        f.close()  
        f = open('./st/paist3_'+str(self.number)+'.txt', 'w')
        f.write(str(self.paist))
        f.close()           

if __name__ == "__main__":
    roads = []
    with open('../data-' + ROAD_SET + '/road_net.json') as f:
        road_net = json.load(f)
    with open('../data-' + ROAD_SET + '/' + ROAD_SET + '_roadSubset', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            roads += row

    dataSet = networkRoadDataset(ROAD_SET)
    batch_len = dataSet.loadDataSet(
        roads, NEIGHBOUR_LEVEL, in_window=INPUT_WIDTH, delay_window=DELAY_WIDTH, out_window=OUTPUT_WIDTH)
    batch_len = 10
    train_break = int(0.6 * batch_len)
    validate_break = train_break + int(0.15 * batch_len)
    idx_train = range(train_break)
    idx_val = range(train_break, validate_break)
    idx_test = range(validate_break, batch_len)

    loader = dataSet.getBatch(roads)

    adj = dataSet.getAdj(roads)
    neighborLoader = dataSet.getNeighborBatch(roads, NUMPY(adj), level=3)

    E = entropys()
    S_train = E.get_pai_s(neighborLoader, idx_train)
    print(S_train)

    '''
    E = entropys()
    S_stTrain = E.get_pai_st(neighborLoader, idx_train)
    S_stVal = E.get_pai_st(neighborLoader, idx_val)
    S_stTest = E.get_pai_st(neighborLoader, idx_test)

    print((S_stTrain[1] + S_stVal[1] + S_stTest[1])/3)
    '''

    '''
    E = entropys()
    Dir = '../data-' + ROAD_SET + '/' + 'TrafficData' + '/'
    paiAct = []
    for road in roads:
        print(road)
        Ds, N = E.count_data(road, Dir)
        ent = E.actualEntropy(torch.tensor([Ds]))
        paiAct.append(E.getPredictability([N], ent))
    print(paiAct)
    '''
