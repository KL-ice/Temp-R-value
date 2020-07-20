import os
import torch
import pandas as pd
import numpy as np
from Entropy import *

'''
get the entropys in Senior way
'''

def split_data(data, n_max=8):
    sp = np.array(list(range(1, n_max+1)))*10/80
    data = np.around(data*80, decimals=-1)/10
    return data

def get_data(name, len_road, batchs):
    all_data = []
    for b in range(batchs):
        x = np.genfromtxt(
            '../out/{}/{}.txt'.format(name, b), dtype=float, delimiter=",")
        all_data.append(x)
    data = np.concatenate(all_data, axis=1)
    data = split_data(data)
    return data

def get_s_tt(data):
    s_tt = []
    N = []
    for r in data:
        count = {}
        n = len(r)-1
        for i in range(n):
            key = str(r[i]) + str(r[i+1])
            if key in count:
                count[key] += 1
            else:
                count[key] = 1
        entropy = 0
        for v in count.values():
            entropy -= (v/n) * math.log(v/n, 2)
        s_tt.append(entropy)
        N.append(len(count.keys()))

    E = Entropy()
    pai_tt = E.get_pai(N, s_tt)

    f = open('./result/s_tt.txt', 'w')
    f.write(str(s_tt))
    f.close()

    f = open('./result/pai_tt.txt', 'w')
    f.write(str(pai_tt))
    f.close()
    return

def get_s_tpt(truth, predict):
    s_tpt = []
    N = []
    for r, pre in zip(truth, predict):
        truth_count = {}
        e_count = {}
        n = len(r)-1
        for i in range(n):
            key = str(r[i]) + str(r[i+1])
            if key in truth_count:
                truth_count[key] += 1
            else:
                truth_count[key] = 1

        entropy = 0
        keys = []
        for i in range(n+1):
            key = str(pre[i]) + str(r[i])
            if key not in keys:
                keys.append(key)
                if key in truth_count:
                    p = truth_count[key]/n
                    entropy -= p * math.log(p, 2)

        s_tpt.append(entropy)
        N.append(len(keys))

    E = Entropy()
    pai_tpt = E.get_pai(N, s_tpt)

    f = open('./result/s_tpt.txt', 'w')
    f.write(str(s_tpt))
    f.close()

    f = open('./result/pai_tpt.txt', 'w')
    f.write(str(pai_tpt))
    f.close()
    return

def get_s_tst(st_loader):
    s_tst = []
    N = []
    print_num = 0
    for r in st_loader:
        print_num += 1
        print(print_num)
        count = {}
        h, w = r.shape
        n = w * (h-1)
        for i in range(w):
            for j in range(h-1):
                key = str(r[j, i]) + str(r[j+1, i])
                if key in count:
                    count[key] += 1
                else:
                    count[key] = 1

        keys = []
        entropy = 0
        for i in range(w):
            for j in range(1, h):
                key = str(r[0, i]) + str(r[j, i])
                if key not in keys:
                    keys.append(key)
                    if key in count:
                        p = count[key] / n
                        entropy -= p * math.log(p, 2)
        s_tst.append(entropy)
        N.append(len(keys))

        E = Entropy()
    pai_tst = E.get_pai(N, s_tst)

    f = open('./result/s_tst.txt', 'w')
    f.write(str(s_tst))
    f.close()

    f = open('./result/pai_tst.txt', 'w')
    f.write(str(pai_tst))
    f.close()
    return

def get_s_tst2(st_loader):
    s_tst2 = []
    N = []
    print_num = 0
    for r in st_loader:
        print_num = print_num + 1
        print(print_num)
        count = {}
        h, w = r.shape
        n = w * (h-1)
        for i in range(w-1):
            for j in range(h-1):
                key = str(r[j, i]) + str(r[j, i+1]) + str(r[j+1, i]) + str(r[j+1, i+1])
                if key in count:
                    count[key] += 1
                else:
                    count[key] = 1

        keys = []
        entropy = 0
        for i in range(w-1):
            for j in range(1, h):
                key = str(r[0, i]) + str(r[0, i+1]) + str(r[j, i]) + str(r[j, i+1])
                if key not in keys:
                    keys.append(key)
                    if key in count:
                        p = count[key] / n
                        entropy -= p * math.log(p, 2)
        s_tst2.append(entropy)
        N.append(len(keys))

        E = Entropy()
    print(s_tst2)
    pai_tst2 = E.get_pai(N, s_tst2)

    f = open('./result/s_tst2.txt', 'w')
    f.write(str(s_tst2))
    f.close()

    f = open('./result/pai_tst2.txt', 'w')
    f.write(str(pai_tst2))
    f.close()
    return

def get_s_row_all(truth):
    E = Entropy()

    _, N = E.get_rand_entropy(truth)
    s_tt = E.get_actual_entropy(truth)
    pai_tt = E.get_pai(N, s_tt)


    f = open('./result/row_all_s.txt', 'w')
    f.write(str(s_tt))
    f.close()

    f = open('./result/row_all_pai.txt', 'w')
    f.write(str(pai_tt))
    f.close()
    return

if __name__ == '__main__':

    t_name = 'truth_qtraffic_all'
    y_name = 'y_qtraffic_all'

    roads = []
    with open('../data-' + ROAD_SET + '/road_net.json') as f:
        road_net = json.load(f)
    with open('../data-' + ROAD_SET + '/' + ROAD_SET + '_roadSubset', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            roads += row
    roads = list(road_net.keys())  # get all road, 7000 roads
    '''
    dataSet = networkRoadDataset(ROAD_SET)
    batch_len = dataSet.loadDataSet(
        roads, NEIGHBOUR_LEVEL, in_window=INPUT_WIDTH, delay_window=DELAY_WIDTH, out_window=OUTPUT_WIDTH)
    adj = dataSet.getAdj(roads)
    st_loader = dataSet.getNeighborBatch(roads, NUMPY(adj), level=4)  # get the st data

    for i in range(len(st_loader)):
        st_loader[i] = split_data(st_loader[i])
        print(st_loader[i].shape)
    
    print('load end')
    get_s_tst2(st_loader)
    print('begin s tst')
    get_s_tst(st_loader)
    
    
    '''
    len_road = len(roads)
    files= os.listdir('../out/{}'.format(t_name))
    len_files = len(files)

    truth = get_data(t_name, len_road, len_files)
    predict = get_data(y_name, len_road, len_files)
    # print(predict)
    print(truth.shape)

    get_s_tt(truth)
    get_s_tpt(truth, predict)
    
    # get_s_row_all(truth[0:2])  # for one road, calculate this with all windows; not use
