# conda install -c conda-forge gensim
# conda install keras

import networkx as nx
import utilities
import node2vec as n2v
import deepwalk as dw
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import methodology
import OLSF 
import curate
import collections
import random


# first line of embedding needs to be removed, dimensions
# dataset needs to be labeled
filepath = '../data/graph_dblp'
emb_path1 = '../embeddings/node2vec.emb'
emb_path2 = '../embeddings/deepwalk.emb'
supergraph_path = '../data/graph_supergraph'


# read and sample the supergraph
g = utilities.read_graph(supergraph_path)
super_subgraph = curate.sample_connected_component(g, 'Xindong Wu', 500)

#alg1 = n2v.node2vec(g,False)
#alg1.learn_embeddings()
#data = utilities.read_embedding(emb_path1)
#alg2 = dw.deepwalk(g,False)
#alg2.learn_embeddings()


# start with drawing a set of publication supernode from super_subgraph:
num_samples = 50
ai_samples = []
db_samples = []

for node in super_subgraph:
    node_obj = super_subgraph.nodes[node]
    if node_obj['type'] == 'publication' and node_obj['label'] == 1 and len(ai_samples)<num_samples:
        ai_samples.append(node_obj)
    if node_obj['type'] == 'publication' and node_obj['label'] == -1 and len(db_samples)<num_samples:
        db_samples.append(node_obj)


samples = ai_samples + db_samples
folds = 1
train_size = int(len(samples) * 0.8)





num_paths = 1  # more data
path_length = 3 # more intersection of nodes



for i in range(1):

    correct = 0
    false = 0
    random.seed(1)
    
    for fold in range(folds):
        w_dict = {}
        w_list = []
        
        random.shuffle(samples)
        train_set = samples[:train_size]
        test_set = samples[train_size:]
        model = OLSF.olsf()
        
        for supernode in train_set:
            y = supernode['label']
            g = supernode['content']
            
            walk_corpus = methodology.build_walk_corpus(g, num_paths, path_length)
            instances, w_dict, w_list = methodology.walk_corpus_to_dataset(walk_corpus, w_dict, w_list)
            
            for instance in instances:
                y_hat = model.predict(instance)
                model.fit(instance, y_hat, y)
        
        for supernode in test_set:
            y = supernode['label']
            g = supernode['content']
            
            walk_corpus = methodology.build_walk_corpus(g, num_paths, path_length)
            instances, d, l = methodology.walk_corpus_to_dataset(walk_corpus, w_dict, w_list)
            # dont update the keys anymore
            pred_count = 0
            
            for instance in instances:
                pred_count += model.predict(instance)
            
            if np.sign(pred_count) == y:
                correct += 1
            else:
                false += 1
                     
    print("\t#Paths, Path Length: "+str(num_paths)+", "+str(path_length))       
    print("\tTest accuracy: "+str((float(correct)/float(correct+false))))
    print()
    path_length+=10
    #num_paths+=1





