''' Implemented following the reference code 
    @ https://github.com/VahidooX/LINE -ege'''
    
# Notes
# After fit_generator, there is a code to test in master file, have a look to make it compatible w/ others
# this takes a long time, probably run it on a lab machine.
    
    

import numpy as np
import random
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers import dot
from keras.models import Sequential, Model
import math
import csv
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import keras.backend as K
import networkx as nx



epoch_num = 100
factors = 128
batch_size = 1000
negative_sampling = "UNIFORM" # UNIFORM or NON-UNIFORM
negativeRatio = 5
split_ratios = [0.6, 0.2, 0.2]
svm_C = 0.1
np.random.seed(2017)
random.seed(2017)
label_file = '../data/line_labels.txt'
edge_file = '../data/line_adjedges.txt'


class line:
    def __init__(self, nx_graph):
        self.generate_LINE_label_list(nx_graph)
        self.generate_LINE_adj_list(nx_graph)
        adj_list, labels_dict = self.load_data(label_file, edge_file)
        epoch_train_size = (((int(len(adj_list)/batch_size))*(1 + negativeRatio)*batch_size) + (1 + negativeRatio)*(len(adj_list)%batch_size))
        numNodes = np.max(adj_list.ravel()) + 1
        data_gen = self.batchgen_train(adj_list, numNodes, batch_size, negativeRatio, negative_sampling)
        model, self.embed_generator = self.create_model(numNodes, factors)
        model.summary()
        model.compile(optimizer='rmsprop', loss={'left_right_dot': self.LINE_loss})
        model.fit_generator(data_gen, samples_per_epoch=epoch_train_size, nb_epoch=epoch_num, verbose=1)
        
        
    def generate_LINE_label_list(self, nx_graph): # get nx graph in, output a label list
        g = nx_graph
        myfile = open(label_file, 'w')
        for node in g:
            label = g.node[node]['class_label']
            node_id = str(node)
            line = node_id + " " + str(label)
            myfile.write("%s\n" % line)
        myfile.close()
        
    
    def generate_LINE_adj_list(self, nx_graph): # get nx graph in, generate adj. list
        nx.write_adjlist(nx_graph, edge_file)

        
    def create_model(self, numNodes, factors):
        left_input = Input(shape=(1,))
        right_input = Input(shape=(1,))

        left_model = Sequential()
        left_model.add(Embedding(input_dim=numNodes + 1, output_dim=factors, input_length=1, mask_zero=False))
        left_model.add(Reshape((factors,)))

        right_model = Sequential()
        right_model.add(Embedding(input_dim=numNodes + 1, output_dim=factors, input_length=1, mask_zero=False))
        right_model.add(Reshape((factors,)))

        left_embed = left_model(left_input)
        right_embed = left_model(right_input)

        left_right_dot = dot([left_embed, right_embed], axes=1, name="left_right_dot")

        model = Model(input=[left_input, right_input], output=[left_right_dot])
        embed_generator = Model(input=[left_input, right_input], output=[left_embed, right_embed])

        return model, embed_generator


    def load_data(self, label_file, edge_file): # fix for nx input
        csvfile = open(label_file, 'r')
        label_data = csv.reader(csvfile, delimiter=' ')
        labels_dict = dict()
        for row in label_data:
            labels_dict[int(row[0])] = int(row[1])

        csvfile = open(edge_file, 'r')
        adj_data = csv.reader(csvfile, delimiter=' ')
        adj_list = None
        for row in adj_data:
            if row[0] == '#': # networkx exports statistics w/ #
                continue
            for j in range(1, len(row)):
                if len(row[j]) == 0:
                    continue
                a = int(row[0])
                b = int(row[j])

                if adj_list is None:
                    adj_list = np.zeros((1, 2), dtype=np.int32)
                    adj_list[0, :] = [a, b]
                else:
                    adj_list = np.concatenate((adj_list, [[a, b]]), axis=0)

        adj_list = np.asarray(adj_list, dtype=np.int32)

        labeler = LabelEncoder()
        labeler.fit(list(set(adj_list.ravel())))

        adj_list = (labeler.transform(adj_list.ravel())).reshape(-1, 2)

        labels_dict = {labeler.transform([k])[0]: v for k, v in labels_dict.items() if k in labeler.classes_}

        return adj_list, labels_dict


    def LINE_loss(self, y_true, y_pred):
        coeff = y_true*2 - 1
        return -K.mean(K.log(K.sigmoid(coeff*y_pred)))


    def batchgen_train(self, adj_list, numNodes, batch_size, negativeRatio, negative_sampling):

        table_size = 1e8
        power = 0.75
        sampling_table = None

        data = np.ones((adj_list.shape[0]), dtype=np.int8)
        mat = csr_matrix((data, (adj_list[:,0], adj_list[:,1])), shape = (numNodes, numNodes), dtype=np.int8)
        batch_size_ones = np.ones((batch_size), dtype=np.int8)

        nb_train_sample = adj_list.shape[0]
        index_array = np.arange(nb_train_sample)

        nb_batch = int(np.ceil(nb_train_sample / float(batch_size)))
        batches = [(i * batch_size, min(nb_train_sample, (i + 1) * batch_size)) for i in range(0, nb_batch)]

        if negative_sampling == "NON-UNIFORM":
            print("Pre-procesing for non-uniform negative sampling!")
            node_degree = np.zeros(numNodes)

            for i in range(len(adj_list)):
                node_degree[adj_list[i,0]] += 1
                node_degree[adj_list[i,1]] += 1

            norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

            sampling_table = np.zeros(int(table_size), dtype=np.uint32)

            p = 0
            i = 0
            for j in range(numNodes):
                p += float(math.pow(node_degree[j], power)) / norm
                while i < table_size and float(i) / table_size < p:
                    sampling_table[i] = j
                    i += 1

        while 1:

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                pos_edge_list = index_array[batch_start:batch_end]
                pos_left_nodes = adj_list[pos_edge_list, 0]
                pos_right_nodes = adj_list[pos_edge_list, 1]

                pos_relation_y = batch_size_ones[0:len(pos_edge_list)]

                neg_left_nodes = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.int32)
                neg_right_nodes = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.int32)

                neg_relation_y = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.int8)

                h = 0
                for i in pos_left_nodes:
                    for k in range(negativeRatio):
                        rn = sampling_table[random.randint(0, table_size - 1)] if negative_sampling == "NON-UNIFORM" else random.randint(0, numNodes - 1)
                        while mat[i, rn] == 1 or i == rn:
                            rn = sampling_table[random.randint(0, table_size - 1)] if negative_sampling == "NON-UNIFORM" else random.randint(0, numNodes - 1)
                        neg_left_nodes[h] = i
                        neg_right_nodes[h] = rn
                        h += 1

                left_nodes = np.concatenate((pos_left_nodes, neg_left_nodes), axis=0)
                right_nodes = np.concatenate((pos_right_nodes, neg_right_nodes), axis=0)
                relation_y = np.concatenate((pos_relation_y, neg_relation_y), axis=0)

                yield ([left_nodes, right_nodes], [relation_y])  