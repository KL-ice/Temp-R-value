import numpy as np

def get_data(name, i):

    name = "./beijing/{}_{}.txt".format(name, i)
    print(name)
    f = open(name, 'r')

    x = []
    lines = f.readlines()
    for i in lines:
        for j in i.split(' '):
            if j == '\n' or j == ']' or j == '[':
                continue
            if j[-1:] == '\n' or  j[-1:] == ']':
                x.append(float(j[:-1]))
            elif j != '':
                if j[0] == '[':
                    x.append(float(j[1:]))
                else:
                    x.append(float(j))
    return np.array(x)

def merge(pools, name):
    x = get_data(name, 0)
    for i in range(1, pools):
        x += get_data(name, i)
    x = x/pools
    f = open('./beijing/all/{}_all.txt'.format(name), 'w')
    f.write(str(x))
    f.close()
    print(x/pools)

if __name__ == '__main__':
    pools = 5
    merge(pools, 't_entropy')
    merge(pools, 't_pai')
    merge(pools, 'st_entropy')
    merge(pools, 'st_pai')