import os
import torch
import pandas as pd
import numpy as np
from Entropy import *
from operateData import *
from multiprocessing import Process, Pool


def get_s_row_all_multi(truth, i):
    """
    cal the row entropy and pai for truth
    :param truth: data, size in (road * speeds)
    :param i: the number of the process
    :return: nothing return, write to file
    """
    print(i)
    E = Entropy()

    rand_e, N = E.get_rand_entropy(truth)
    s_tt = E.get_actual_entropy(truth)
    # unc_e = E.get_unc_entropy(truth)
    print('process ', i, 'cal all entropy end')
    pai_tt = E.get_pai(N, s_tt)
    # rand_pai = E.get_pai(N, rand_e)
    # unc_pai = E.get_pai(N, unc_e)

    np.savetxt('./result_{}_row/row_all_actE/{}.txt'.format(ROAD_SET, i), s_tt)
    np.savetxt('./result_{}_row/row_all_actpai/{}.txt'.format(ROAD_SET, i), pai_tt)

    # np.savetxt('./result_{}_row/row_all_randE/{}.txt'.format(ROAD_SET, i), rand_e)
    # np.savetxt('./result_{}_row/row_all_randpai/{}.txt'.format(ROAD_SET, i), rand_pai)

    # np.savetxt('./result_{}_row/row_all_uncE/{}.txt'.format(ROAD_SET, i), unc_e)
    # np.savetxt('./result_{}_row/row_all_uncpai/{}.txt'.format(ROAD_SET, i), unc_pai)

    return


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

    pai_tt = np.array(pai_tt)
    s_tt = np.array(s_tt)

    np.savetxt('./result_{}_row/actE.txt'.format(ROAD_SET), s_tt)
    np.savetxt('./result_{}_row/actpai.txt'.format(ROAD_SET), pai_tt)

    return


if __name__ == '__main__':
    pools = 3  # 13 for pems
    t_name = 'truth_qtraffic_all'

    roads = []
    with open('../data-' + ROAD_SET + '/road_net.json') as f:
        road_net = json.load(f)
    with open('../data-' + ROAD_SET + '/' + ROAD_SET + '_roadSubset', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            roads += row
    roads = list(road_net.keys())  # get all road, 7000 roads

    len_road = len(roads)
    files = os.listdir('../out/{}'.format(t_name))
    len_files = len(files)

    truth = get_data(t_name, len_road, len_files)
    _, w = truth.shape
    truth = truth[:, 0:int(0.6*w)]
    print(truth.shape)

    # get_s_tt(truth)


    truth = np.array_split(truth, 25)
    i_list = [12,19,2] # [0,1,2,8,9,10,11,12,18,19,21,22,24]  # pems_last
    # i_list = [18,19,20,21,22,23,24] # [10,11,12,13,14,15,16,17]# [5,6,7,8,9]  # [0,1,2,3,4]  # qtraffic begin
    # creat pools
    p = Pool(pools)
    for i in i_list:
        print(truth[i].shape)
        p.apply_async(get_s_row_all_multi, args=(truth[i], i))

    # begin process
    print('Waiting for all subprocesses done...')

    # waiting for done
    p.close()
    p.join()
    print('All subprocess done')


