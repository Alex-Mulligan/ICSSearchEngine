'''
Created on Feb 22, 2020

@author: ajm00
'''

import re, glob, time, json, sys
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from queue import PriorityQueue
from nltk.stem import PorterStemmer
from urllib.parse import urldefrag, urljoin, urlsplit
from math import log
import hashlib, networkx
from PageRank import run_pagerank

class Posting:
    def __init__(self, docID, freq, positions=[]):
        self.docID = docID
        self.freq = freq
        self.positions = positions
    def __lt__(self, right):
        return self.docID < right.docID
    def __gt__(self, right):
        return self.docID > right.docID
    def __str__(self):
        return '({}, {})'.format(self.docID, self.freq)
    def __eq__(self, right):
        return self.docID == right.docID

class Indexer:
    def __init__(self):
        self.start = time.time() #times the index construction
        self.index = defaultdict(PriorityQueue)
        self.docDict = dict()
        self.docs_to_ids = dict()
        self.urls_visited = set()
        self.sites = glob.iglob('DEV/**/*.json', recursive=True)
        self.docID = 0
        self.urls = 0
        self.index_number = 0
        self.anchors = defaultdict(list)
        self.postitions = defaultdict(list)
        self.doc_words = []
        self.simhashes = []
        self.graph = networkx.Graph()
        self.site_titles = dict() #docID -> site title
        self.temp_title = tuple()
    
    def run(self):
        for i in self.sites:
            if sys.getsizeof(self.index) > 5242880: #5 mB
                self.sort_and_write(self.index, self.index_number)
                self.index.clear()
                self.index_number += 1
            with open(i, 'r') as file:
                file_contents = file.read()
                f = json.loads(file_contents)
                if urldefrag(f['url']).url in self.urls_visited:
                    continue
                page_text = self.extract_text(f['content'], urldefrag(f['url']).url)
                stems = self.gather_tokens(page_text)
                if not self.simhash():
                    continue
                self.docID += 1
                self.docDict[self.docID] = urldefrag(f['url']).url
                self.docs_to_ids[urldefrag(f['url']).url] = self.docID
                self.urls_visited.add(urldefrag(f['url']).url)
                if self.temp_title:
                    self.site_titles[self.temp_title[0]] = self.temp_title[1]
                freq_count = Counter(stems) #counts the frequencies
                stems = set(stems) #removes duplicates
                for s in stems:
                    self.index[s].put_nowait(Posting(self.docID, freq_count[s]))
            self.urls += 1
            print(self.urls)
        if self.index:
            self.sort_and_write(self.index, self.index_number)
        else:
            self.index_number -= 1
        self.count_anchor_text()
        self.merge_indexes_and_write(self.index_number, self.urls)
        self.write_urls(self.docDict)
        self.make_bookkeeping()
        self.make_titles()
        print(f"There were {self.urls} urls")
        #print(f"There were {len(self.index)} unique tokens")
        print(f"It took {time.time() - self.start} seconds")
        print("Running PageRank...")
        run_pagerank()
        print("Done.")
        
    
    def extract_text(self, page, url):
        soup = BeautifulSoup(page, features='lxml')
        #blacklist = ['[document]','noscript','header','html','meta','head','input','script','footer','div','style','i','li','g','section']
        whitelist = set(['p', 'span','title','h1','h2','h3','h4','h5','h6','strong','b']) #span,p
        page_text = ''
        ps = PorterStemmer()
        self.temp_title = tuple()
        for w in soup.findAll(text=True):
            if w.parent.name in whitelist:
                if w.parent.name == 'title':
                    weight = 3
                    if not self.temp_title:
                        self.temp_title = (self.docID + 1, w)
                elif w.parent.name == 'p' or 'span':
                    weight = 1
                else:
                    weight = 2
                for _ in range(0, weight):
                    page_text += "{} ".format(w)
                    
        out_links_set = set()
        for a in soup('a', href=True):
            at= []
            if a['href'] and a['href'][0] != '#':
                page_text += "{} ".format(a.text.strip())
                link = urldefrag(urljoin(url, a['href'])).url
                if re.match(r".*\.uci\.edu", urlsplit(link).netloc):
                    out_links_set.add(link)
                    at = re.findall(r"[a-zA-Z][a-zA-Z0-9]*|\d{1,4}", a.text.lower())
                    for t in set(at):
                        self.anchors[ps.stem(t)].append(link)
        
        out_degree = len(out_links_set)
        for l in out_links_set:
            self.graph.add_edge(url, l, weight=1.0/out_degree)
        #page_text = soup.get_text(separator=" ")
        return page_text
    
    def gather_tokens(self, text):
        regex = re.compile(r"[a-zA-Z][a-zA-Z0-9]*|\d{1,4}")
        tokens = []
        ps = PorterStemmer()
#         tokens.extend([ps.stem(w) for w in regex.findall(text.lower())])
#         prev_token = ''
        self.doc_words = []
        for w in regex.findall(text.lower()):
            self.doc_words.append(w)
            tokens.append(ps.stem(w))
            #append a 2-gram of the previous token and current token
#             if prev_token:
#                 tokens.append((prev_token, w))
#                 prev_token = w
        return tokens
    
    def sort_and_write(self, index, n):
        with open('index{}.txt'.format(n), 'a+', encoding='utf-8') as file:
            keys = sorted([k for k in index.keys()])
            for k in keys:
                postingList = ''
                doc_freq = 0
                while not index[k].empty():
                    postingList += str(index[k].get_nowait()) +';'
                    doc_freq += 1
                file.write(f'{k}#{doc_freq}->{postingList.rstrip(";")}\n')
    
    def merge_indexes_and_write(self, n, docs):
        files = dict() #map int -> file
        indexes = defaultdict(dict) #map int -> token -> postings
        tokens = defaultdict(list) #int -> list(token)
        frequencies = defaultdict(int)
        for i in range(0,n+1):
            exec(f'index_{i} = open("index{i}.txt", "r", encoding="utf-8")')
            exec(f'files[{i}] = index_{i}')
        files[n+1] = open('anchors.txt', 'r', encoding='utf-8')
        for i in range(0,n+2):
            entries = [x.strip() for x in files[i].readlines(1048576)] #reads 1 mB
            for e in entries:
                k,f,v = self.parse_line(e)
                indexes[i][k] = v
                tokens[i].append(k)
                frequencies[k] += eval(f)
            tokens[i].reverse()
        write_buffer = ''
        while True:
            stop = True
            for i in range(0, n+2): #checks to make sure there is still an open file
                if not files[i].closed:
                    stop = False
            if stop:
                break
            for i,t in tokens.items():
                if not files[i].closed and len(t) == 0: #checks to make sure there are tokens
                    entries = [x.strip() for x in files[i].readlines(1048576)] #reads the next chuck (1 mB) into memory
                    if not entries:
                        files[i].close() #if it got nothing, close the file
                        continue
                    for e in entries:
                        k,f,v = self.parse_line(e)
                        indexes[i][k] = v
                        tokens[i].append(k)
                        frequencies[k] += eval(f)
                    tokens[i].reverse()
            t = [tokens[i][-1] for i in range(0,n+2) if tokens[i]] #gets all the current tokens
            m = min(t) if t else None
            if m:
                posting_buffer = ''
                for i in range(0, n+2):
                    if files[i].closed:
                        continue
                    if tokens[i][-1] == m:
                        tokens[i].pop()
                        posting_buffer += indexes[i].pop(m) + ','
                write_buffer += f"{m}#{frequencies[m]}->{self.translate_postings(posting_buffer.rstrip(','), frequencies[m], docs)}\n"
            if sys.getsizeof(write_buffer) > 10485760: #10mB
                with open('index.txt', 'a+', encoding='utf-8') as out_file:
                    out_file.write(write_buffer)
                write_buffer = ''
        with open('index.txt', 'a+', encoding='utf-8') as out_file:
            out_file.write(write_buffer)
        
    def parse_line(self, l):
        token, parse_list = re.split(r"->",l)
        term,freq = re.split(r'#', token)
        return term, freq, parse_list
    
    def translate_postings(self, p, doc_freq, docs):
        p2 = re.sub(r";", ',', p)
        l = eval(f"[{p2}]")
        l2 = [(a,self.calculate_tfidf(b,doc_freq,docs)) for a,b in l]
        outs = ''
        for t,i in l2:
            outs += '(%s,%.4f);' %(t,i)
        return outs.rstrip(';')
            
    def calculate_tfidf(self, term_freq, doc_freq, num_docs):
        tf = 1 + log(term_freq, 10)
        idf = log(num_docs/doc_freq,10)
        return tf * idf
    
    def write_urls(self, u):
        with open('urls.txt', 'w') as file:
            json.dump(u, file)
    
    def make_bookkeeping(self):
        offset = 0
        book = dict()
        with open('index.txt', 'r', encoding='utf-8') as index:
            for line in index:
                k,f,v = self.parse_line(line)
                book[k] = offset
                offset += len(line)+1
        with open('bookkeeping.txt', 'w', encoding='utf-8') as file:
            json.dump(book, file)
            
    def make_titles(self):
        with open('titles.txt', 'w', encoding='utf-8') as file:
            json.dump(self.site_titles, file)
    
    def simhash(self): #hash in python is 64 bits
        v = [0]*64
        mask = [1 << i for i in range(0,64)]
        for term, freq in Counter(self.doc_words).items():
            term_hash = hash(term)
            for i in range(0,64):
                v[i] += freq if term_hash & mask[i] else -freq
        s = 0
        for i in range(0, 64):
            s |= mask[i] if v[i] > 0 else 0
        
        #adding simhash
        for d in self.simhashes:
            if bin(s ^ d).count('1') < 1:
                return False
        self.simhashes.append(s)
        return True
    
    def count_anchor_text(self):
        j = defaultdict(PriorityQueue)
        for k,v in self.anchors.items():
            for u in v:
                try:
                    j[k].put_nowait(Posting(self.docs_to_ids[u], 1))
                except:
                    pass
        with open('anchors.txt', 'a+', encoding='utf-8') as a:
            keys = sorted([k for k in j.keys()])
            for k in keys:
                postingList = ''
                doc_freq = 0
                while not j[k].empty():
                    postingList += str(j[k].get_nowait()) +';'
                    doc_freq += 1
                if postingList.strip() != "":
                    a.write(f'{k}#{doc_freq}->{postingList.rstrip(";")}\n')
        
    
if __name__ == '__main__':
    I = Indexer()
    I.run()
    