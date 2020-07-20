import torch
import pandas
import numpy as np
from Entropy import *
from multiprocessing import Process, Pool

def get_data(name, len_road, batchs):
    all_data = []
    for b in range(batchs):
        x = np.genfromtxt('../out/{}/{}.txt'.format(name, b), dtype=float, delimiter=",")
        all_data.append(x)
    data = np.concatenate(all_data , axis=1)
    return data

def split_data(data, n_max=8):
    sp = np.array(list(range(1, n_max+1)))*10/80
    data = np.around(data*80, decimals=-1)/10
    return data

def get_format_data(truth, predict, need_truth = False, need_predict = False):
    h, w = truth.shape
    max_row = int((w - in_window) / out_window)
    X = []
    for i in range(max_row):
        x = truth[:, i*out_window:i*out_window + in_window]
        if need_truth:
            t = truth[:, (i+1)*in_window:(i+1)*in_window + out_window]
            x = np.concatenate((x, t), axis=1)
        if need_predict:
            t = predict[:, (i+1)*in_window:(i+1)*in_window + out_window]
            x = np.concatenate((x, t), axis=1)
        X.append(x)
    return X

def get_another_data(truth, predict):
    h, w = truth.shape
    max_row = int((w - in_window) / out_window)
    X = []
    print(w, max_row)
    for i in range(max_row):
        x = []
        y = truth[:, i*out_window:i*out_window + in_window]
        p = predict[:, i*out_window:i*out_window + in_window]
        p_next = predict[:,i*out_window + in_window]
        k=0
        for yi, pi, pni in zip(y, p, p_next):
            k+=1
            roadi = []
            for yij, pij in zip(yi, pi):
                if pij in [pni, pni+1, pni-1]:
                    roadi.append(yij)

            if len(roadi) == 0:
                roadi = yi

            x.append(roadi)        
        X.append(x)
    return X

def get_s_err(truth, predict, i):
    err = truth - predict
    err = get_format_data(err, None, False, False)
    # print(len(err))
    
    E = Entropy()
    err_E, err_pai = [], []
    print_num = 0
    for e in err:
        print_num += 1
        if print_num % 100 == 0:
            print(i, print_num)
        _, N = E.get_rand_entropy(e)
        err_E.append(E.get_actual_entropy(e))
        err_pai.append(E.get_pai(N, err_E[-1]))
        # print(N)
    
    err_E = np.mean(np.array(err_E), axis=0)
    err_pai = np.mean(np.array(err_pai), axis=0)

    np.savetxt('./result/err_E/{}.txt'.format(i), err_E)
    np.savetxt('./result/err_pai/{}.txt'.format(i), err_pai)

    return

def get_my_idea(truth, predict):
    X = get_another_data(truth, predict)

    E = Entropy()
    load_pre, err_pai = [], []
    N_all = []
    print_num = 0
    for r in X:
        print_num += 1
        print(print_num)
        _, N = E.get_rand_entropy(r)
        load_pre.append(E.get_actual_entropy(r))
        err_pai.append(E.get_pai(N, load_pre[-1]))
        print(N[-2])
        N_all.append(N)
    
    load_pre = np.mean(np.array(load_pre), axis=0)
    load_pre_pai = np.mean(np.array(err_pai), axis=0)
    N_all = np.mean(np.array(N_all), axis=0)

    f = open('./result/load_pre.txt', 'w')
    f.write(str(load_pre))
    f.close()

    f = open('./result/load_pre_pai.txt', 'w')
    f.write(str(load_pre_pai))
    f.close()

    f = open('./result/load_pai_N_all.txt', 'w')
    f.write(str(N_all))
    f.close()

    return

def get_row_s(truth, predict):
    X = get_format_data(truth, predict, False, False)

    E = Entropy()
    row_s, row_pai = [], []
    N_all = []
    print_num = 0
    print(len(X))
    for r in X:
        print_num += 1
        print(print_num)
        _, N = E.get_rand_entropy(r)
        row_s.append(E.get_actual_entropy(r))
        row_pai.append(E.get_pai(N, row_s[-1]))

        print(N)
        N_all.append(N)
    
    row_s = np.mean(np.array(row_s), axis=0)
    row_pai = np.mean(np.array(row_pai), axis=0)
    N_all = np.mean(np.array(N_all), axis=0)

    f = open('./result/row_s.txt', 'w')
    f.write(str(row_s))
    f.close()

    f = open('./result/row_pai.txt', 'w')
    f.write(str(row_pai))
    f.close()

    f = open('./result/row_N_all.txt', 'w')
    f.write(str(N_all))
    f.close()

    return

in_window  = 96
out_window = 8

if __name__ == '__main__':
    pools = 25
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
    len_road = len(roads)  # get the number 7555
    files = os.listdir('../out/{}'.format(t_name))
    batchs = len(files)
    truth = get_data(t_name, len_road, batchs)
    truth = split_data(truth) 
    predict = get_data(y_name, len_road, batchs)
    predict = split_data(predict)
    print(truth.shape)
    
    truth = np.array_split(truth, pools)
    predict = np.array_split(predict, pools)

    # creat pools
    p = Pool(pools)
    for i in range(pools):
        print(truth[i].shape)
        p.apply_async(get_s_err, args=(truth[i], predict[i], i))

    # begin process
    print('Waiting for all subprocesses done...')

    # waiting for done
    p.close()
    p.join()
    print('All subprocess done')

    # get_s_err(truth, predict)

    # get_my_idea(truth, predict)

    # get_row_s(truth, predict)  # for one road, calculate this with one window, and cal average of all windows
