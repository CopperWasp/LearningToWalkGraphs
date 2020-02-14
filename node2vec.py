''' Implemented following the Py2 reference code 
    @ https://github.com/aditya-grover/node2vec  -ege'''
    
    # 1. graph needs to have a ['weight'] field
    
import numpy as np
#import networkx as nx
import random
from gensim.models import Word2Vec


# Hyperparameters
is_directed = False
p = 1
q = 1
num_walks = 10
walk_length = 5
dimensions = 3
window_size = 10
workers = 8
iter = 1
output = '../embeddings/node2vec.emb'


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        
        
    def node2vec_walk(self, walk_length, start_node): # what is alias draw 
        #print('node2vec_walk called.')
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        
        while len(walk) < walk_length:
            cur = walk[-1] # current node
            cur_nbrs = sorted(G.neighbors(cur)) # nx utility
            
            if len(cur_nbrs) > 0:
                if len(walk) == 1: # no prev
                    a0 = alias_nodes[cur][0]
                    a1 = alias_nodes[cur][1]
                    walk.append(cur_nbrs[self.alias_draw(a0, a1)])
                    
                else:
                    prev = walk[-2]
                    a0 = alias_edges[(prev, cur)][0]
                    a1 = alias_edges[(prev, cur)][1]
                    next = cur_nbrs[self.alias_draw(a0, a1)]
                    walk.append(next)
                    
            else:
                break
            
        return walk
    
    
    def simulate_walks(self, num_walks, walk_length):
        #print('simulate_walks called.')
        # repeatedly simulate random walks from each node.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes) # start node is randomly changing, so not every node get to be start
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        
        return walks
    
    
    def get_alias_edge(self, src, dst):
        #print('get_alias_edge called')
        # get the alias edge setup lists for a given edge.
        G = self.G
        p = self.p
        q = self.q
        
        unnormalized_probs = []
        
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
                
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        return self.alias_setup(normalized_probs)
    

    def preprocess_transition_probs(self):
        #print('process_transition_probs called')
        G = self.G
        is_directed = self.is_directed
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return


    def alias_setup(self, probs):

        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)
        
        smaller = []
        larger = []
        
        for kk,prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
                
        
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
                
        return J, q
    
    
    def alias_draw(self, J, q):
        K = len(J)
        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
    
                
class node2vec():
    def __init__(self, G, isWeighted):
        if isWeighted == False:
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1
                
        self.G = G
            
    def learn_embeddings(self):
        print('node2vec is learning embeddings.')
        graph = Graph(self.G, is_directed, p, q)
        graph.preprocess_transition_probs()
        walks = graph.simulate_walks(num_walks, walk_length)
        walks = list([list(map(str,walk)) for walk in walks])
        model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=iter)
        model.wv.save_word2vec_format(output)
        print('node2vec embeddings saved to '+str(output)+'.')
        

        
        
        
        
        
        

