'''
Created on Feb 28, 2020

@author: ajm00
'''

import re, time, json
from Indexer import Posting
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from math import log10, sqrt

class Engine():
    def __init__(self):
        self.url_dict = dict() #int -> url
        self.index = open('index.txt', 'r', encoding='utf-8') #term -> list(Posting)
        self.read_url_dict()
        self.bookkeeping = dict()
        self.read_bookkeeping()
        self.page_ranks = dict()
        self.read_pagerank()
        self.cache = dict() #cache the latest search terms and posting lists
        self.titles = dict()
        self.read_titles()
        self.num_urls = len(self.url_dict)
        
    def read_url_dict(self):
#         s = re.compile(r"->")
        with open('urls.txt', 'r') as file:
#             for line in file:
#                 docID,url = s.split(line)
#                 self.url_dict[docID] = url.strip()
            self.url_dict = json.load(file)#, object_hook=lambda x: {int(k) if k.isdigit() else k : v for k,v in x.items()})
    
    def read_bookkeeping(self):
        with open('bookkeeping.txt', 'r') as file:
            self.bookkeeping = json.load(file)
    
    def read_index(self): #this was just for testing, not actual use
        with open('index000.txt', 'r') as file:
            for line in file:
                term, post = re.split(r"->", line)
                pl = [Posting(docID, freq) for (docID,freq) in [eval(t) for t in re.split(';', post)]]
                t = re.split(r"#", term)
                self.index[t[0]] = pl
    
    def read_pagerank(self):
        with open('pagerank.txt', 'r') as file:
            for line in file:
                docID,rank = line.split(';')
                self.page_ranks[docID] = rank
    
    def read_titles(self):
        with open('titles.txt', 'r') as file:
            self.titles = json.load(file)
    
    def parse_index_line(self, term):
        s = self.bookkeeping[term]
        self.index.seek(s)
        line = self.index.readline().strip()
        t,p = re.split(r"->", line)
        pl = [Posting(docID, freq) for (docID,freq) in [eval(t) for t in re.split(';', p) if t]]
        return pl
    
    def get_df(self, term):
        s = self.bookkeeping[term]
        self.index.seek(s)
        line = self.index.readline().strip()
        t,p = re.split(r"->", line)
        word,df = re.split(r"#", t)
        return eval(df)
    
    def bool_search(self, query):
        #terms = query.split()
        terms = query
        #used to be sorted([{p.docID for p in self.index[t]} for t in terms], key=lambda x: len(x), reverse=True)
        try:
#             ids = sorted([sorted([p for p in self.index[t]], reverse=True) for t in terms], key=lambda x: len(x), reverse=True)
            ids = sorted([sorted([p for p in self.parse_index_line(t)], reverse=True) for t in terms], key=lambda x: len(x), reverse=True)
        except KeyError:
            return []
#         possible = ids.pop()
        possible = []
        term1_list = ids.pop()
        if(len(ids)) == 0:
            possible = [(p.docID, p.freq) for p in term1_list]
        while ids:
            term2_list = ids.pop()
            t1 = term1_list.pop()
            t2 = term2_list.pop()
            while term1_list and term2_list:
                if t1 == t2:
                    possible.append((t1.docID, t1.freq + t2.freq))
                    t1 = term1_list.pop()
                    t2 = term2_list.pop()
                elif t1 < t2:
                    t1 = term1_list.pop()
                else:
                    t2 = term2_list.pop()
            term1_list = [Posting(k,v) for k,v in possible]
            if not term1_list:
                break
        
        return sorted(possible, key=lambda x: x[1], reverse=True)
    
    def rank_search(self, query):
        q = Counter(query)
        tfidf = dict()
        for t,v in q.items():
            tfidf[t] = (1 + log10(v)) * log10(self.num_urls/(self.get_df(t)+1)) if t in self.bookkeeping.keys() else 1
        
        query = list(set(query))
        A = dict()
        L = []
        query_order = []
        for t in query:
            query_order.append(t)
            if t in self.cache.keys():
                L.append([i for i in self.cache[t]])
            else:
                if t in self.bookkeeping.keys():
                    x = sorted([p for p in self.parse_index_line(t)], reverse=True)
                    self.cache[t] = [i for i in x]
                    L.append(x)
                
        #L = sorted([sorted([p for p in self.parse_index_line(t)], reverse=True) for t in query if t in self.bookkeeping.keys()], key=lambda x: len(x))
        i = 0
        for pl in L:
            while pl:
                p = pl.pop()
                if p.docID not in A.keys():
                    A[p.docID] = 0
                A[p.docID] += p.freq * tfidf[query_order[i]]
            i += 1
            
        for k in A.keys():
            if k in self.page_ranks.keys():
                A[k] += sqrt(self.page_ranks[k])
        
        return sorted([s for s in A.items()], key=lambda x: x[1] , reverse=True)
        
if __name__ == '__main__':
    e = Engine()
#     e.read_index()
    ps = PorterStemmer()
    search_terms = ['cristina lopes','machine learning','ACM','master of software engineering']
    for s in search_terms:
        start = time.time()
        query = [ps.stem(w) for w in s.split()]
        options = e.rank_search(query)
        print(s)
        for i in range(0, min(len(options), 5)):
            print(e.url_dict[str(options[i][0])], options[i][1])
        print(time.time() - start)
        