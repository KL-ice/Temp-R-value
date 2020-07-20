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
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F

from dLoader import *
from config import *
from method import *

class Entropy():

    def __init__(self, t_loader = None, st_loader = None, number = 0, n_max = 8):
        self.t_loader = t_loader
        self.st_loader = st_loader
        self.number = number
        self.n_max = n_max

        self.rand_entropy, self.unc_entropy, self.t_entropy, self.s_entropy,  self.st_entropy= [], [], [], [], []
        self.rand_pai, self.unc_pai, self.t_pai, self.s_pai, self.st_pai = [], [], [], [], []

    def get_rand_entropy(self, l):
        """
        get the rand entropy
        input: l: road * in_window
        """
        randE = []
        N = []
        for i in l:
            i = np.around(i, decimals=2)
            count = pd.value_counts(i)
            n = len(count)
            if n == 0:
                print(l)
            S_rand = math.log(n, 2)
            randE.append(S_rand)
            N.append(n)
        return randE, N

    def get_unc_entropy(self, l):
        '''
        get the unc entropy
        input: l: road * in_window
        '''
        uncE = []
        for i in l:
            i = np.around(i, decimals=2)
            count = pd.value_counts(i).as_matrix()
            n = np.sum(count)
            count = count/n * np.log2(count/n)
            S_unc = -np.sum(count)
            uncE.append(S_unc)
        return uncE

    def t_contains(self, small, big):
        '''
        Determine if the small is in the big
        '''
        for i in range(len(big)-len(small)+1):
            if big[i:i+len(small)] == small:
                return True
        return False

    def get_actual_entropy(self, l):
        '''
        get the actual entropy
        input: l: road * in_window
        '''
        num = 0
        actE = []
        for k in l:
            num += 1
            print(num)
            '''保留n位小数'''
            k = np.around(k, decimals=2)
            n = len(k)
            sequence = [k[0]]
            sum_gamma = 0

            for i in range(1, n):
                appear = True
                for j in range(i+1, n+1):
                    s = list(k[i:j])
                    if self.t_contains(s, sequence) != True:
                        sum_gamma += len(s)
                        sequence.append(k[i])
                        appear = False
                        break
                if appear:
                    sum_gamma += n - i + 1
            if sum_gamma == 0:
                sum_gamma  = 1
            ae = 1 / (sum_gamma / n) * math.log(n, 2)
            actE.append(ae)
        return actE

    def st_contains(self, small, big, h):
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

    def one_road_ste(self, l):
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
                    if not self.st_contains(s, sequence, i):
                        sum_gamma += k
                        break
                sequence[i].append(l[i, j])

        if sum_gamma == 0 or n == 0:
            ae = 0
        else:
            ae = 1 / (sum_gamma / n) * math.log(n, 2)
        ni = np.unique(l).shape[0]
        return ae, ni

    def get_st_entropy(self, l):
        '''
        get the st entropy
        input: l: road * neighbor * in_window
        '''
        actE = []
        N = []
        for k in l:
            '''保留n位小数, k: neighbor * in_window'''
            k = np.around(k, decimals=2)
            e, n = self.one_road_ste(k)
            actE.append(e)
            N.append(n)
        return actE, N

    def get_pai(self, N, S):
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

    def get_s_entropy(self, l):
        '''
        get the s entropy
        input: l: road * neighbor * time
        '''
        S_s = []
        pai_s = []
        for i in l:    # i -> one road
            i = i.T
            S = self.get_actual_entropy(i)   #96*1
            _, N = self.get_rand_entropy(i)  #96*1
            pai = self.get_pai(N, S)  #96*1

            S_s.append(np.mean(S))
            pai_s.append(np.mean(pai))
        return S_s, pai_s
            
    def cal_entropy_t(self):
        '''
        get the entropy and pai for data without s information [loader = getBatch()]
        input: batch * road * time
        '''
        rand_E, unc_E, t_E = [], [], []
        rand_pai, unc_pai, t_pai = [], [], []

        i = 0
        for x in self.t_loader:
            print('t name:{} NO.{}'.format(self.number, i))
            i = i + 1

            rand_entropy, N = self.get_rand_entropy(x)
            rand_E.append(rand_entropy)
            #unc_E.append(self.get_unc_entropy(x))
            t_E.append(self.get_actual_entropy(x))

            #rand_pai.append(self.get_pai(N, rand_E[-1]))
            #unc_pai.append(self.get_pai(N, unc_E[-1]))
            t_pai.append(self.get_pai(N, t_E[-1]))

        #self.rand_entropy = np.mean(np.array(rand_E), axis=0)
        #self.rand_pai = np.mean(np.array(rand_pai), axis=0)

        #self.unc_entropy = np.mean(np.array(unc_E), axis=0)
        #self.unc_pai = np.mean(np.array(unc_pai), axis=0)

        self.t_entropy = np.mean(np.array(t_E), axis=0)
        self.t_pai = np.mean(np.array(t_pai), axis=0)

        return

    def cal_entropy_st(self):
        '''
        get the entropy and pai for data with s information [loader = getImageBatch()]
        input: batch * road * neighbor * time
        '''
        s_entropy, st_entropy = [], []
        s_pai, st_pai = [], []

        i = 0
        for x in self.st_loader:
            print('st name:{} NO.{}'.format(self.number, i))
            i = i + 1
            
            sst, N = self.get_st_entropy(x)
            st_entropy.append(sst)
            print(N)
            st_pai.append(self.get_pai(N, st_entropy[-1]))

            #ss, sp = self.s_entropy()
            #s_entropy.append(ss)
            #s_pai.append(sp)

        #self.s_entropy = np.mean(np.array(s_entropy), axis=0)
        #self.s_pai = np.mean(np.array(s_pai), axis=0)
        self.st_entropy = np.mean(np.array(st_entropy), axis=0)
        self.st_pai = np.mean(np.array(st_pai), axis=0)
        

        return

if __name__ == '__main__':
    roads = []
    with open('../data-' + ROAD_SET + '/' + ROAD_SET + '_roadSubset', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            roads += row
    print(roads)
    dataSet = networkRoadDataset(ROAD_SET)
    batch_len = dataSet.loadDataSet(roads, NEIGHBOUR_LEVEL, in_window = INPUT_WIDTH, delay_window = DELAY_WIDTH, out_window = OUTPUT_WIDTH)
    batch_len = 10
    train_break = int(0.6 * batch_len)
    validate_break = train_break + int(0.15 * batch_len)
    idx_train = range(train_break)

    t_loader = dataSet.getBatch(roads)
    adj = dataSet.getAdj(roads)
    st_loader = dataSet.getNeighborBatch(roads, NUMPY(adj), level=3)

    E = Entropy(t_loader, st_loader, idx_train)
    # E.cal_entropy_t()
    E.cal_entropy_st()
    print(E.t_pai)
    print(E.st_pai)
