# data set homepage (v10): https://aminer.org/citation
# nx utility methods: https://networkx.github.io/documentation/stable/reference/classes/graph.html
# dynamic plot: https://stackoverflow.com/questions/13437284/animating-network-growth-with-networkx-and-matplotlib

import json
import networkx as nx
import glob
from collections import Counter
import re
import math
import time
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import collections
import matplotlib.pyplot as plt


WORD = re.compile(r'\w+')
path = '../data/dblp*.json'
filename = 'dblp'


def read_files(path):
    nodes = []
    for file in glob.glob(path):
        print('Reading from: '+str(file))
        for line in open(file, 'r'):
            nodes.append(json.loads(line))
    print("readToDict done.")
    return nodes


def read_title_info():
    with open('../data/ai_titles.txt') as f:
        ai_titles = f.read().splitlines()
    with open('../data/db_titles.txt') as f:
        db_titles = f.read().splitlines()
    return ai_titles, db_titles


def process_data(nodes): # remove the ones w/ missing attributes
    ai_titles, db_titles = read_title_info()
    cleaned_nodes = [node for node in nodes 
                             if (('abstract' in node.keys()) 
                            and ('references' in node.keys())
                            and ('venue' in node.keys())
                            and (node['venue'] in ai_titles+db_titles))
                            ]
    print("processData removed the entries without the fields 'abstract', 'references' and 'venue'.")
    print("processData removed the entries published in venues different than the provided lists.")
    
    for row in cleaned_nodes:
        if row['venue'] in ai_titles:
            row['label'] = 1
        elif row['venue'] in db_titles:
            row['label'] = -1
        else:
            print('unknown venue.')
    
    return cleaned_nodes
            

def generate_network(nodes):
    g = nx.Graph()
    ai_titles, db_titles = read_title_info()
    for node in nodes: # first add all nodes to avoid any override while adding edges
        if node['venue'] in ai_titles:
            label = 1
        elif node['venue'] in db_titles:
            label = -1
        else:
            print("something's wrong with the process_data method.")
        g.add_node(node['id'], abstract=node['abstract'], venue=node['venue'], class_label=label, ref=node['references']) # duplicates don't make change, you can add nodes multiple times, won't add if exists
        
    for node in g: # only add the edges that refer to existing nodes, otherwise you will be adding nodes that are out of our standards.
        for ref in g.node[node]['ref']:
            if ref in g:
                 g.add_edge(node, ref)
                

    print("generateNetwork done, network info:")
    print(nx.info(g))
    timestamp = str(time.strftime("%Y%m%d-%H%M%S"))
    #save_graph(g, timestamp)
    save_graph(g, filename)
    print("Graph is saved as a gPickle file under the ../data directory, with the timestamp "+str(timestamp)+".")
    g = nx.convert_node_labels_to_integers(g)
    return g


def generate_raw_network(nodes): # all data for statistics
    g = nx.Graph()
    for node in nodes:
        g.add_node(node['id'])
        if 'references' in node.keys():
            for ref in node['references']:
                g.add_edge(node['id'], ref)
    return g
        

def get_stats(data):
    title_check = []
    venue_counts = Counter(tok['venue'] for tok in data)
    ai_titles, db_titles = read_title_info()
    for title in ai_titles:
        title_check.append((title, True if title in venue_counts.keys() else False))
    for title in db_titles:
        title_check.append((title, True if title in venue_counts.keys() else False))  
    print(title_check)
    return title_check, venue_counts


def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])
     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator


def text_to_vector(text):
    return Counter(WORD.findall(text))


def get_similar_titles(data):
    data_titles = Counter(tok['venue'] for tok in data).keys()
    ai_titles, db_titles = read_title_info()
    text_titles = ai_titles + db_titles
    for title1 in text_titles:
        max_cosine = 0
        max_cosine_title = ''
        for title2 in data_titles:
            vector1 = text_to_vector(title1)
            vector2 = text_to_vector(title2)
            cosine = get_cosine(vector1, vector2)
            if cosine>max_cosine: 
                max_cosine = cosine
                max_cosine_title = title2
        print(title1+" best match: "+max_cosine_title+" with cos. sim. of "+str(max_cosine))
        

def search_regex(keyword, data):
    string = '.*'+keyword+'.*'
    matches = list(filter(re.compile(string).match, data))
    print(matches)
        
    
def save_graph(g, name):
    nx.write_gpickle(g, '../data/graph_'+name)


def read_graph(filename):
    return nx.read_gpickle(filename)


def iterate_nodes(g):
    for node in g:
        print(g.node[node].keys())
        

def remove_node(g, node_id): # Removes the node n and all adjacent edges. Attempting to remove a non-existent node will raise an exception.
    g.remove_node(node_id)
    

def get_nth_node_object(g, n): #
    return g.node[list(g.nodes.keys())[n]]


def get_nth_node_id(g, n):
    return list(g.nodes.keys())[n]


def set_nth_node_attribute(g, n): # example for future
    handle = get_nth_node_id(g,n)
    g.node[handle]['graph'] = []
    
    
def read_embedding(emb_path):
    with open(emb_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [x.split() for x in content]
    content = [{x[0]:np.array([float(val) for val in x[1:]])} for x in content]
    return content


def svm_classify(X, label, split_ratios, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor for SVM
    """
    train_size = int(len(X)*split_ratios[0])
    val_size = int(len(X)*split_ratios[1])

    train_data, valid_data, test_data = X[0:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    train_label, valid_label, test_label = label[0:train_size], label[train_size:train_size + val_size], label[train_size + val_size:]

    print('training SVM...')
    clf = svm.SVC(C=C, kernel='linear')
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(train_data)
    train_acc = accuracy_score(train_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)
    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)

    return [train_acc, valid_acc, test_acc]
    

#data = process_data(read_files(path))
#g = generate_network(data)
#save_graph(g, 'graph_dblp')

def plot_degree_dist(G): # log scaled because heavy tailed
    degrees = [G.degree(n) for n in G.nodes()]
    plt.xlabel('Num. nodes')
    plt.ylabel('Degree (log)')
    plt.hist(degrees, log=True)
    plt.savefig('./degree_dist.png', dpi=300)
    
    
def get_total_authors(data):
    a_dict = {}
    for row in data:
        if 'authors' in row.keys():
            for author in row['authors']:
                if author in a_dict.keys():
                    a_dict[author]+=1
                else:
                    a_dict[author]=1
    print(len(a_dict.keys()))
    
    
def get_total_venues(data):
    a_dict = {}
    for row in data:
        if 'venue' in row.keys():
            if row['venue'] not in a_dict.keys():
                a_dict[row['venue']] = 1
    print(len(a_dict.keys()))
    