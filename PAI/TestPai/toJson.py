import json
from Entropy import *


def cat_data():
    """
    cat the data from every process
    :return: data with size (7555, )
    """
    data = []
    for i in range(25):
        d = np.loadtxt('./result/err_pai/{}.txt'.format(i))
        data.append(d)
    data = np.concatenate(data, axis=0)
    return data


def cat_row_data():
    """
    cat the data from every process
    :return: data with size (7555, )
    """
    data = []
    names = ['row_all_actE', 'row_all_actpai', 'row_all_uncE', 'row_all_uncpai', 'row_all_randE', 'row_all_randpai']
    for n in names:
        data_son = []
        for i in range(25):
            d = np.loadtxt('./result_beijing_row/{}/{}.txt'.format(n, i))
            data_son.append(d)
        data_son = np.concatenate(data_son, axis=0)
        data.append(data_son)
        
    data = np.stack(data, axis=1)
    
    return data


def get_all_data():
    """
    get the pai from txt then cal mean and var
    :rtype: data: numpy array, roads * information(pai, mean, var)
    """
    names = ['pai_tt', 'pai_tpt', 'pai_tst', 'pai_tst2']
    data = []
    d = cat_data()
    data.append(d)
    for name in names:
        d = np.genfromtxt('./result/{}.txt'.format(name), delimiter=',')
        data.append(d)
        print(d.shape)

    data = np.stack(data, axis=1)
    mean = np.mean(data, axis=1)
    var = np.var(data, axis=1)
    cal = np.stack([mean, var], axis=1)
    
    res = np.concatenate((data, cal), axis=1)
    return res


def to_json(d, r):
    """
    write the data to json
    :rtype: nothing return
    :param d: data
    :param r: road
    :return:
    """
    dic = {}
    for i in range(len(r)):
        dic[r[i]] = d[i].tolist()

    with open(".//pais_beijing_row_all.json", "w") as f:
        json.dump(dic, f)
    return

if __name__ == '__main__':
    roads = []
    with open('../data-' + ROAD_SET + '/road_net.json') as f:
        road_net = json.load(f)
    with open('../data-' + ROAD_SET + '/' + ROAD_SET + '_roadSubset', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            roads += row
    roads = list(road_net.keys())  # get all road, 7000 roads
    print(len(roads))
    # data = get_all_data()
    data = cat_row_data()
    print(data.shape)

    to_json(data, roads)
