import networkx as nx
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
from itertools import combinations
import matplotlib.pyplot as plt
import utilities
import random


def plot_weighted(g): # uncomment for better visualization
    plt.figure(figsize=(18, 18))
    pos=nx.spring_layout(g, iterations=5) # spectral_layout is also informative
    nx.draw(g,pos,with_labels=True, width=0.2)
    labels = nx.get_edge_attributes(g,'weight')
    nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)


def tokenize_sentence(sentence):
    tokens = word_tokenize(sentence)
    lower_tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in lower_tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words


def tokenize_paragraph(paragraph):
    tokens = sent_tokenize(paragraph)
    return tokens
    
    
class supernode:
    def __init__(self, obj_type, obj_id, graph):
        self.content = graph
        self.id = obj_id
        self.type = obj_type
        self.paper_id = None
        self.label = None
        

def generate_author_object(author_name, data):
    g = nx.Graph()
    for paper in data:
        if author_name in paper['authors']:
            title = paper['title']
            tokens = tokenize_sentence(title)
            g.add_nodes_from(tokens)
    
            for pair in combinations(tokens, 2):            
                t1 = pair[0]
                t2 = pair[1]
        
                if t1 != t2:
                    if g.has_edge(t1, t2):
                        g[t1][t2]['weight'] += 1
                    else:
                        g.add_edge(t1, t2, weight=1)
                        
    author = supernode('author', author_name, g)
    return author


def generate_publication_object(index, data):
    g = nx.Graph()
    abstract = data[index]['abstract']
    sentences = tokenize_paragraph(abstract)
    for sentence in sentences:
        tokens = tokenize_sentence(sentence)
        g.add_nodes_from(tokens)
        
        for pair in combinations(tokens, 2):            
                t1 = pair[0]
                t2 = pair[1]
        
                if t1 != t2:
                    if g.has_edge(t1, t2):
                        g[t1][t2]['weight'] += 1
                    else:
                        g.add_edge(t1, t2, weight=1)
                        
    publication = supernode('publication', data[index]['title'], g)
    publication.paper_id = data[index]['id']
    publication.label = data[index]['label']
    return publication


def generate_venue_object(venue_name, data):
    g = nx.Graph()
    citation_dict = {}
    count_dict = {}
    for paper in data:
        if venue_name == paper['venue']:
            year = paper['year']
            citations = paper['n_citation']
            
            if year in citation_dict.keys():
                citation_dict[year] += citations
                count_dict[year] += 1
            else:
                citation_dict[year] = citations
                count_dict[year] = 1
                
    for year in citation_dict.keys():
        g.add_node(year, weight=(float(citation_dict[year])/count_dict[year]))
    
    venue = supernode('venue', venue_name, g)
    return venue


def generate_supergraph(data):
    g = nx.Graph()
    
    # create a dict of id-title for faster execution
    id_title = {}
    for row in data:
        id_title[row['id']] = row['title']
        
    for i in range(len(data)):
        if(i % 50 == 0):
            print(str(i)+'/'+str(len(data)), end='\r')
        
        paper = data[i]
        # paper
        p_obj = generate_publication_object(i, data)
        v_obj = generate_venue_object(paper['venue'], data)
        a_obj_list = []
        
        g.add_node(p_obj.id, content=p_obj.content, type=p_obj.type, label=p_obj.label)
        g.add_node(v_obj.id, content=v_obj.content, type=v_obj.type, label=p_obj.label)
        
        g.add_edge(p_obj.id, v_obj.id) # add paper - venue edge
        
        # generate author component
        for author in paper['authors']:
            a_obj = generate_author_object(author, data)
            a_obj_list.append(a_obj)
            g.add_node(a_obj.id, content=a_obj.content, type=a_obj.type, label=p_obj.label)
            g.add_edge(a_obj.id, p_obj.id) # add author - paper edges
            
        # add paper-paper edge        
        for paper_id in paper['references']: # inefficient but its a one-time job
            title2 = id_title[paper_id]
            g.add_edge(paper['title'], title2)
    
    utilities.save_graph(g, 'supergraph')      
    return g
                    
                    
def remove_outside_references(data):
    print('\nRunning remove_outside_references.')
    all_ids = {}
    all_refs = []
    zero_citation_counter = 0
    
    for row in data:
        all_ids[row['id']] = False
        all_refs += row['references']
        if row['n_citation'] == 0: zero_citation_counter+=1
    
    
    intersection = list(set.intersection(set(all_ids.keys()), set(all_refs)))
    print('Size of the intersect. set of references and publication objects: '+str(len(intersection)))
    all_refs = []
     
    # this is for removing outside references
    for row in data:
        references = row['references']
        new_references = []
        for r in references:
            if r in all_ids.keys():
                new_references.append(r)
        row['references'] = new_references
        all_refs += row['references']
        
        
    # this is for checking if any work with citations are referenced by an outer publication:
    for row in data:
        references = row['references']
        for ref in references:
            all_ids[ref] = True
            
    outside_reference_counter = 0
    for row in data:
        if row['n_citation'] != 0 and all_ids[row['id']] == False:
            # then this work is referred by an outside publication
            outside_reference_counter +=1
        
    print('Number of unique publications: '+str(len(set(all_ids.keys()))))
    print('Number of unique referenced publications after removing outside references: '+str(len(set(all_refs))))
    print('Number of 0 citation publications: '+str(zero_citation_counter))
    print('Number of non-zero citation pubs. referred by outside publications: '+str(outside_reference_counter))
    check_bit = (outside_reference_counter + zero_citation_counter + len(set(all_refs))) == len(set(all_ids.keys()))
    print('Check if the summations hold: '+str(check_bit))
    if check_bit == True:
        print('Therefore, there are publication objects that are cited by outside publications.')
    else:
        print('Something is wrong with the reference removing strategy.')

    return data
        
        
 
def print_supergraph_statistics(g):
    g.name = 'DBLP Supergraph'
    print(nx.info(g))
    num_authors = 0
    num_publications = 0
    num_venues = 0
    
    s_pp = set(['publication','publication'])
    s_pa = set(['publication','author'])
    s_pv = set(['publication','venue'])
    
    pp = 0
    pa = 0
    pv = 0
    
    for node in g:
        content = g.node[node]['content']
        label = g.node[node]['label']
        typ = g.node[node]['type']
        
        if typ == 'author':
            num_authors += 1
        elif typ == 'publication': 
            num_publications += 1
        elif typ == 'venue': 
            num_venues += 1
        else: 
            print('Node' + str(node)+ ' has unknown type.')
            
    print('Num. author objects: '+str(num_authors))
    print('Num. venue objects: '+str(num_venues))
    print('Num. publication objects: '+str(num_publications))
        

    for edge in g.edges:
        #print(edge)
        s = set([
                (g.node[edge[0]])['type'], 
                (g.node[edge[1]])['type']
                ])
        if s == s_pp:
            pp+=1
        elif s == s_pa:
            pa+=1
        elif s == s_pv:
            pv+=1
        else:
            print('PROBLEM.')
            
    print('Num. p-p edges: '+str(pp))
    print('Num. p-v edges: '+str(pv))
    print('Num. p-a edges: '+str(pa))
            
    
        
        
def sample_connected_component(g, start, size):
    d = dict(nx.bfs_successors(g, source=start))
    return g.subgraph(list(d.keys())[:size])
   

            
# add labels
# check the clean-data method

# generate the graph and test
                    
        
        
        # venue
        
        # author




#data = remove_outside_references(utilities.process_data(utilities.read_files(utilities.path)))
#g = utilities.generate_network(data)


        
        
        