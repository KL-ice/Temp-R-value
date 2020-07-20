import numpy as np 
import json
import torch
import os
import random
import sys
import csv
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset

class networkRoadDataset():
    def __init__(self, dataSet):
        self.dataDir = '../data-' + dataSet + '/TrafficData/'
        self.roadNetFile = '../data-' + dataSet + '/road_net.json'
        self.roadSet = set()
        for f in os.listdir(self.dataDir):  
            self.roadSet.add(f)
        with open(self.roadNetFile, 'r') as f:
            self.roadNet = json.load(f)
    
    def __fetchData__(self):
        roads = list(self.roadDataSet.keys())
        len_road = len(roads)

        y_start = self.in_window + self.delay_window
        y_end = y_start + self.max_row * self.out_window

        x_end = y_end - self.out_window - self.delay_window
        for i in range(len_road):
            road = roads[i]
            entity = self.roadDataSet[road]
            if not entity['loaded']:
                x_data = entity['x'][:y_end]
                #y_max = max(1, np.max(x_data[:int(0.6 * self.max_row)]))
                y_max = 80
                x_data /= y_max
                x_data[x_data > 1.] = 1.
                y_mean = 0
                #y_mean = np.mean(x_data)
                x_data -= y_mean
                entity.update({
                    'x': x_data,
                    'y_max': y_max,
                    'y_mean': y_mean,
                    'loaded': True,
                })
                log = "Normalizing: %d"%(i + 1) + "/%d"%len_road
                print(log, end = "")
                print('\b'*len(log), end = "", flush=True)
            else:
                return #assert all data is loaded if one road data is loaded
        print("\n>>> All data batches fetched, ready for training")

    def __scanRoads__(self):
        max_row = self.max_row
        road_len = len(list(self.roadDataSet.items()))
        count = 0
        size = 0
        for road, entity in self.roadDataSet.items():
            count += 1
            x_file = self.dataDir + road
            tp_x_data = np.genfromtxt(x_file, delimiter=' ')
            max_row = min(((tp_x_data.shape[0] - self.in_window - self.delay_window) // self.out_window), max_row)
            entity.update({'x': tp_x_data, 'loaded': False})
            size += sys.getsizeof(tp_x_data)
            log = "Scanning: %d"%count + "/%d"%road_len + ", current road: " + road + ", total size: %dbytes"%(size)
            print(log, end = "")
            print('\b'*len(log), end = "", flush=True)
        print()
        self.max_row = max_row
    
    def __scanNeighboursDirection__(self, baseRoads, existRoads, prev = False):
        label = 'd'
        if prev:
            label = 'o'
        _len = len(baseRoads)
        newRoads = []
        for i in range(_len):
            road = baseRoads[i]
            if road in self.roadNet:
                neighbourList = self.roadNet[road][label]
                for j in range(len(neighbourList)):
                    neighbour = neighbourList[j]
                    if neighbour not in existRoads:
                        existRoads.add(neighbour)
                        newRoads.append(neighbour)
                    if neighbour not in self.roadDataSet:
                        self.roadDataSet[neighbour] = {}
                        #print('load new road ' + neighbour)
        return newRoads
    
    def __scanNeighbours__(self, baseRoads, existRoads, prevF = True, nextF = True):
        newRoads = {
            'next': self.__scanNeighboursDirection__(baseRoads['next'], existRoads) if nextF else [],
            'prev': self.__scanNeighboursDirection__(baseRoads['prev'], existRoads, True) if prevF else []
        }
        return newRoads

    def loadDataSet(self, roads, max_neighbour_level = 1, in_window = 96, delay_window = 0, out_window = 8):
        len_road = len(roads)
        self.roadDataSet = {}
        self.roadNeighbour = {}
        self.max_row = sys.maxsize
        self.level = max_neighbour_level

        [self.in_window, self.delay_window, self.out_window] = [in_window, delay_window, out_window]

        for i in range(len_road):
            road = roads[i]
            if road not in self.roadSet:
                print("[WARNING]Skip nonexistent dataset file " + road)
                continue
            else:
                self.roadDataSet[road] = {}
                
        for i in range(len_road):
            road = roads[i]
            if self.level:
                roadChain = [road]
                baseRoads = {
                    'prev': [road],
                    'next': [road]
                }
                curr_level = 0
                while curr_level < self.level:
                    baseRoads = self.__scanNeighbours__(baseRoads, set(roadChain))
                    roadChain = roadChain + baseRoads['prev'] + baseRoads['next']
                    curr_level += 1
                self.roadNeighbour[road] = roadChain

        print('>>> Graph Anaylsis completed, collect %d'%(len(list(self.roadDataSet.keys()))) + ' roads')
        self.__scanRoads__()
        return self.max_row
        #print(self.max_row)

    def __getAdj__(self, roads):
        roadRef = {}
        edges = []
        for i in range(len(roads)):
            road = roads[i]
            if road not in self.roadDataSet:
                raise Exception("Unexcepted road " + road)
            roadRef[road] = i

        for i in range(len(roads)):
            road = roads[i]
            neighbourList = self.roadNet[road]['d']
            src = roadRef[road]
            for j in range(len(neighbourList)):
                neighbour = neighbourList[j]
                if neighbour in roadRef:
                    dst = roadRef[neighbour]
                    edges.append([src, dst])
        road_len = len(roads)
        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(road_len, road_len), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        print(">>> Adjacency matrix builded, find %d"%(edges.shape[0]) + " edges")
        return (torch.FloatTensor(np.array(adj.todense())), roadRef)

    def getAdj(self, roads):
        adj, refs = self.__getAdj__(roads)
        return adj

    def getNearBatch(self, roads):
        self.__fetchData__()
        road_len = len(roads)
        _max = []
        _mean = []

        _train_rows = int(0.6 * self.max_row)
        _validate_rows = _train_rows + int(0.15 * self.max_row)
        _test_rows = self.max_row - _validate_rows
        roadDataSet = {}
        for k in range(road_len):
            entity = self.roadDataSet[roads[k]]
            _max.append(entity['y_max'])
            _mean.append(entity['y_mean'])
            road = roads[k]
            x_tp = []
            roadChain = self.roadNeighbour[road][:self.level]
            if len(roadChain) < self.level:
                raise Exception("Road chain too short: " + ",".join(roadChain))
            len_roads = len(roadChain)
            _arr = []
            for i in range(len_roads):
                neighbour = self.roadDataSet[roadChain[i]]
                neighbour_renorm = (neighbour['x'] + neighbour['y_mean']) * neighbour['y_max']
                neighbour_renorm = neighbour_renorm / entity['y_max'] - entity['y_mean']
                x_tp.append(neighbour_renorm)
            roadDataSet[roads[k]] = np.stack(x_tp, 0)
            #print(roadDataSet[roads[k]].shape)
        
        _max = torch.from_numpy(np.array(_max)).view(-1, 1)
        _mean = torch.from_numpy(np.array(_mean)).view(-1, 1)
        for j in range(self.max_row):
            _x_start = j * self.out_window
            _x_end = _x_start + self.in_window
            _y_start = _x_end + self.delay_window
            _y_end = _y_start + self.out_window
            _x = []
            _y = []
            for k in range(road_len):
                x_data = roadDataSet[roads[k]]
                #x_image = torch.from_numpy(x_data[:, _x_start:_x_end]).view(1, -1, self.in_window)
                #_x.append(x_image)
                _x.append(x_data[:, _x_start:_x_end]) #[roads, h, w]
                _y.append(x_data[0, _y_start:_y_end])
            yield (torch.from_numpy(np.stack(_x, 0)), torch.from_numpy(np.stack(_y, 0)), _max, _mean)
        
    def getImageBatch(self, roads, adj, level = 1):
        self.__fetchData__()
        road_len = len(roads)
        _max = []
        _mean = []
        new_adj = adj.copy()
        new_adj[new_adj <= 0] = np.nan

        _train_rows = int(0.6 * self.max_row)
        _validate_rows = _train_rows + int(0.15 * self.max_row)
        _test_rows = self.max_row - _validate_rows
        roadDataSet = {}
        for k in range(road_len):
            entity = self.roadDataSet[roads[k]]
            _max.append(entity['y_max'])
            _mean.append(entity['y_mean'])
            x_tp = [entity['x']]
            last_index = set()
            index = k
            for j in range(level):
                last_index.add(index)
                if np.isnan(np.nanstd(new_adj[index])):
                    #print(new_adj[index])
                    break
                elif np.nanstd(new_adj[index]) >= 0:
                    index = np.nanargmax(new_adj[index])
                else:
                    #print(np.nanstd(new_adj[index]))
                    break
                if index in last_index:
                    break
                neighbour = self.roadDataSet[roads[index]]
                neighbour_renorm = (neighbour['x'] + neighbour['y_mean']) * neighbour['y_max']
                neighbour_renorm = neighbour_renorm / entity['y_max'] - entity['y_mean']
                x_tp.append(neighbour_renorm)
                
            roadDataSet[roads[k]] = np.stack(x_tp, 0)
            #print(roadDataSet[roads[k]].shape)
        
        _max = torch.from_numpy(np.array(_max)).view(-1, 1)
        _mean = torch.from_numpy(np.array(_mean)).view(-1, 1)
        for j in range(self.max_row):
            _x_start = j * self.out_window
            _x_end = _x_start + self.in_window
            _y_start = _x_end + self.delay_window
            _y_end = _y_start + self.out_window
            _x = []
            _y = []
            for k in range(road_len):
                x_data = roadDataSet[roads[k]]
                x_image = torch.from_numpy(x_data[:, _x_start:_x_end]).view(1, 1, -1, self.in_window)
                _x.append(x_image)
                #_x.append(x_data[:, _x_start:_x_end]) #[roads, h, w]
                _y.append(x_data[0, _y_start:_y_end])
            yield (_x, torch.from_numpy(np.stack(_y, 0)), _max, _mean)

    def getNeighborBatch(self, roads, adj, level=1, num=2776):
        self.__fetchData__()
        road_len = len(roads)
        new_adj = adj.copy()
        new_adj[new_adj <= 0] = 0

        roadDataSet = {}
        for k in range(road_len):
            entity = self.roadDataSet[roads[k]]
            x_tp = [entity['x'][-num:]]
            last_index = set()
            index = set([k])

            for j in range(level):
                last_index = last_index | index
                neighbors = set()
                for idx in index:
                    if np.sum(new_adj[idx]) == 0:
                        break
                    elif np.nanstd(new_adj[idx]) >= 0:
                        nb = np.nonzero(new_adj[idx])
                        neighbors = neighbors | set(nb[0])  #0 : get the list
                    else:
                        break
                index = neighbors - last_index #this level new find 
                if not index:
                    break
                for idx in index:
                    neighbour = self.roadDataSet[roads[idx]]['x'][-num:]
                    x_tp.append(neighbour)

            roadDataSet[roads[k]] = np.stack(x_tp, 0)
        
        all_data = []

        _x = []
        for k in range(road_len):
            x_data = roadDataSet[roads[k]]
            _x.append(x_data)

        return _x

    def getLastBatch(self, roads):
        self.__fetchData__()

        road_len = len(roads)
        _max = []
        _mean = []

        _train_rows = int(0.6 * self.max_row)
        _validate_rows = _train_rows + int(0.15 * self.max_row)
        _test_rows = self.max_row - _validate_rows

        for k in range(road_len):
            entity = self.roadDataSet[roads[k]]
            _max.append(entity['y_max'])
            _mean.append(entity['y_mean'])
        
        _max = torch.from_numpy(np.array(_max)).view(-1, 1)
        _mean = torch.from_numpy(np.array(_mean)).view(-1, 1)
        _x_start = (self.max_row - 1) * self.out_window
        _x_end = _x_start + self.in_window
        _y_start = _x_end + self.delay_window
        _y_end = _y_start + self.out_window
        _x = []
        _y = []
        for k in range(road_len):
            x_data = self.roadDataSet[roads[k]]['x']
            _x.append(x_data[_x_start:_x_end])
            _y.append(x_data[_y_start:_y_end])
        return (torch.from_numpy(np.stack(_x, 0)), torch.from_numpy(np.stack(_y, 0)))

    def getBatch(self, roads):
        self.__fetchData__()

        road_len = len(roads)
        _max = []
        _mean = []

        _train_rows = int(0.6 * self.max_row)
        _validate_rows = _train_rows + int(0.15 * self.max_row)
        _test_rows = self.max_row - _validate_rows

        for k in range(road_len):
            entity = self.roadDataSet[roads[k]]
            _max.append(entity['y_max'])
            _mean.append(entity['y_mean'])
        
        _max = torch.from_numpy(np.array(_max)).view(-1, 1)
        _mean = torch.from_numpy(np.array(_mean)).view(-1, 1)

        all_data = []
        for j in range(self.max_row):
            _x_start = j * self.out_window
            _x_end = _x_start + self.in_window
            _y_start = _x_end + self.delay_window
            _y_end = _y_start + self.out_window
            _x = []
            _y = []
            for k in range(road_len):
                x_data = self.roadDataSet[roads[k]]['x']
                _x.append(x_data[_x_start:_x_end])
                _y.append(x_data[_y_start:_y_end])
            #yield (torch.from_numpy(np.stack(_x, 0)), torch.from_numpy(np.stack(_y, 0)), _max, _mean)
            all_data.append(_x)
        return all_data

if __name__ == "__main__":
    loader = networkRoadDataset('qtraffic')
    roads = []
    with open('../data-qtraffic/road_net.json') as f:
        road_net = json.load(f)
    road_arr = road_net.keys()
    with open('../data-qtraffic/qtraffic_roadSubset', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            roads += row
    #print(roads)
    loader.loadDataSet(roads, 1, in_window = 96, delay_window = 0, out_window = 2)
    batch_loader = loader.getBatch()
    (batch, adj) = next(batch_loader)
    print('input shape:', end = '')
    print(batch[0].shape)
    print('output shape:', end = '')
    print(batch[1].shape)
    print('label shape:', end = '')
    print(batch[2].shape)
    print('Adjacency matrix shape:', end = '')
    print(adj.shape)
    #loader, _max = loader.getXYWithNeighbour('595672_30202', 64)
    #step, (x, y) = list(enumerate(loader['train']))[0]
    #print(x.shape)
    #print(y.shape)
    #print(x[0])
    #print(x[1])
    #print(_max)
