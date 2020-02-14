''' Implemented following the reference code 
    @ https://github.com/phanein/deepwalk  -ege'''
    
import random
from gensim.models import Word2Vec
from six import iterkeys
from collections import defaultdict, Iterable
    
    
output = '../embeddings/deepwalk.emb'
is_undirected = True
num_walks = 10
walk_length = 5
seed = 1
dimensions = 3
window_size = 10
workers = 8



class deepwalk():
    def __init__(self, G, isWeighted):
        if isWeighted == False:
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1
                
        self.G = self.from_networkx(G)
        #self.num_walks = len(G.nodes()) * num_walks # for stats.
        #self.data_size = self.num_walks * walk_length # for stats.
        
            
    def learn_embeddings(self):
        print('Deepwalk is learning embeddings.')
        walks = self.build_deepwalk_corpus(self.G, num_paths=num_walks, path_length=walk_length, alpha=0, rand=random.Random(seed))
        model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, hs=1, workers=workers)
        model.wv.save_word2vec_format(output)
        print('Deepwalk embeddings saved to '+str(output)+'.')
        
        
    def build_deepwalk_corpus(self, G, num_paths, path_length, alpha=0, rand=random.Random(0)):
        walks = []
        nodes = list(G.nodes())
          
        for cnt in range(num_paths):
            rand.shuffle(nodes)
            for node in nodes:
                walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
          
        return walks
  
    
    def from_networkx(self, G_input): # read into Graph format
        G = Graph()
        for idx, x in enumerate(G_input.nodes):
            for y in iterkeys(G_input[x]):
                G[x].append(y)
    
        return G
    
    
class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)
    

  def nodes(self):
    return self.keys()


  def adjacency_iter(self):
    return self.iteritems()


  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]
        
    return subgraph


  def make_undirected(self):
    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    self.make_consistent()
    return self


  def make_consistent(self):
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    self.remove_self_loops()
    return self


  def remove_self_loops(self):
    removed = 0
    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    return self


  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False


  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False


  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])


  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    


  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2


  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order()


  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]




