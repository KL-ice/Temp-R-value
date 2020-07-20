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
    unc_e = E.get_unc_entropy(truth)
    print('process ', i, 'cal all entropy end')
    pai_tt = E.get_pai(N, s_tt)
    rand_pai = E.get_pai(N, rand_e)
    unc_pai = E.get_pai(N, unc_e)

    np.savetxt('./result_beijing_row/row_all_actE/{}.txt'.format(i), s_tt)
    np.savetxt('./result_beijing_row/row_all_actpai/{}.txt'.format(i), pai_tt)

    np.savetxt('./result_beijing_row/row_all_randE/{}.txt'.format(i), rand_e)
    np.savetxt('./result_beijing_row/row_all_randpai/{}.txt'.format(i), rand_pai)

    np.savetxt('./result_beijing_row/row_all_uncE/{}.txt'.format(i), unc_e)
    np.savetxt('./result_beijing_row/row_all_uncpai/{}.txt'.format(i), unc_pai)

    return


if __name__ == '__main__':
    pools = 25
    t_name = 'truth_beijing'

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
    print(truth.shape)
    _, w = truth.shape

    truth = np.array_split(truth[:, 0:int(0.6*w)], pools)

    # creat pools
    p = Pool(3)
    for i in [0,1,7]:
        print(truth[i].shape)
        p.apply_async(get_s_row_all_multi, args=(truth[i], i))

    # begin process
    print('Waiting for all subprocesses done...')

    # waiting for done
    p.close()
    p.join()
    print('All subprocess done')


