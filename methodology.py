import random
import collections
import numpy as np

def random_walk(G, path_length, alpha=0, rand=random.Random(), start=None): # won't leave the supernode
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.nodes))]
    
    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(list(G.neighbors(cur))))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]


def build_walk_corpus(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    # paths per node
    walks = []
    nodes = list(G.nodes())
      
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(G, path_length, rand=rand, alpha=alpha, start=node))
      
    return walks


def walk_to_instance(G, walk): # for a single walk
    # get a random walk, return it as a feature-value dict
    # designed to work on node- not supernode
    inst = {}
    for node in walk:
        inst[node] = 1
    return inst



def walk_corpus_to_dataset(walk_corpus, prev_dict, prev_list): # keys are from previous objects
    words_dict = {} # fast lookup
    words_list = prev_list # ordered
    
    for walk in walk_corpus:
        for word in walk:
            if word not in words_dict.keys():
                words_dict[word] = 1
                words_list.append(word)
            else:
                words_dict[word] += 1
                
    instances = []

    for walk in walk_corpus:
        instance = []
        for word in words_list:
            if word in walk:
                instance.append(1)
            else:
                instance.append(0)
            
        instances.append(np.array(instance))
    return instances, words_dict, words_list