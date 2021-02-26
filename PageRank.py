'''
Created on Mar 12, 2020

@author: ajm00
'''

from __future__ import division
from numba import cuda, float32
import numpy, math, networkx, json, scipy, glob
from collections import defaultdict
from urllib.parse import urldefrag, urlsplit, urljoin
from bs4 import BeautifulSoup

# CUDA kernel
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp
        
@cuda.jit
def matadd(A,B,C):
    """Perform matrix addition of C = A + B
    """
    row,col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row,col] = A[row,col] + B[row,col]
    
@cuda.jit   
def scalarmul(A, B, constant):
    """Perform scalar multiplication of constant * A = B
    """
    row,col = cuda.grid(2)
    if row < B.shape[0] and col < B.shape[1]:
        B[row,col] = A[row,col] * constant
        
#Actual pagerank function        
        
def compute_PageRank(graph, u2i):
    """Computes the pageranks for a given graph.
    Uses a damping factor to make the matrix irreducible.
    PR(k+1) = a*M*PR(k) + (1-a)*PR(1)
    """
    damping = 0.85
    urls = [n for n in graph.nodes]
    num_urls = len(urls)
    site_matrix = networkx.to_scipy_sparse_matrix(graph, dtype=numpy.float32)
    temp = defaultdict(list)
    for i in range(len(site_matrix)):
        for j in range(len(site_matrix[i])):
            temp[j].append(site_matrix[i][j])
    site_matrix = numpy.array([x for x in temp.values()], numpy.float32)
    
    ranks = numpy.ones((num_urls, 1), numpy.float32)
    #ranks = numpy.ones((1, num_urls), numpy.float32)
    damping_matrix = numpy.full((num_urls, 1), (1-damping), numpy.float32)
    #damping_matrix = numpy.full((1, num_urls), (1-damping), numpy.float32)
    damping_global_mem = cuda.to_device(damping_matrix)
    site_global_mem = cuda.to_device(site_matrix)
    
    threadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(site_matrix.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(ranks.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    for _ in range(6):
        rank_global_mem = cuda.to_device(ranks)
        product_global_mem = cuda.device_array((num_urls, 1))
        #product_global_mem = cuda.device_array((1, num_urls))
        matmul[blockspergrid, threadsperblock](site_global_mem, rank_global_mem, product_global_mem)
        scaled_global_mem = cuda.device_array((num_urls, 1))
        #scaled_global_mem = cuda.device_array((1, num_urls))
        scalarmul[blockspergrid, threadsperblock](product_global_mem, scaled_global_mem, damping)
        sum_global_mem = cuda.device_array((num_urls, 1))
        #sum_global_mem = cuda.device_array((1, num_urls))
        matadd[blockspergrid, threadsperblock](scaled_global_mem, damping_global_mem, sum_global_mem)
        ans = sum_global_mem.copy_to_host()
        ranks = ans
    
    pageranks = dict()
    for x in range(num_urls):
        try:
            pageranks[u2i[urls[x]]] = ranks[x][0]
        except:
            continue
    
    with open('pagerank.txt', 'w', encoding='utf-8') as file:
        json.dump(pageranks, file)
        
    
def helper_PageRank(graph, u2i):
    damping = .85
    urls = [n for n in graph.nodes]
    num_urls = len(urls)
    site_matrix = networkx.to_scipy_sparse_matrix(graph, dtype=numpy.float32)
    r,c,d = scipy.sparse.find(site_matrix)
    site_matrix = scipy.sparse.csr_matrix((d, (c,r)))
    ranks = numpy.ones((num_urls, 1), numpy.float32)
    damping_matrix = numpy.full((num_urls, 1), (1-damping), numpy.float32)
    
    for _ in range(100):
        ranks = site_matrix.dot(ranks)
        ranks = ranks * damping
        ranks = ranks + damping_matrix
    
    pageranks = dict()
    for x in range(num_urls):
        try:
            pageranks[u2i[urls[x]]] = ranks[x][0]
        except:
            continue
    with open('pagerank.txt', 'w', encoding='utf-8') as file:
        for k,v in pageranks.items():
            file.write(f"{k};{v}\n")
            
def run_pagerank():
    print("Prefiltering...")
    sites = glob.iglob('DEV/**/*.json', recursive=True)
    urls_visited = set()
    valid_urls = set()
    x = 0
    for i in sites:
        with open(i, 'r') as file:
            f = json.loads(file.read())
            url = urldefrag(f['url']).url
            if url in urls_visited:
                continue
            urls_visited.add(url)
            valid_urls.add(url)
            x += 1
        if x % 1000 == 0:
            print(x)
    sites = glob.iglob('DEV/**/*.json', recursive=True)
    urls_visited = set()
    g = networkx.DiGraph()
    docDict = dict()
    docID = 0
    print("Done.")
    
    print("Creating Graph...")
    for i in sites:
        with open(i, 'r') as file:
            file_contents = file.read()
            f = json.loads(file_contents)
            if urldefrag(f['url']).url in urls_visited:
                continue
             
            docID += 1
            url = urldefrag(f['url']).url
            urls_visited.add(url)
            docDict[url] = docID
            soup = BeautifulSoup(f['content'], features="lxml")
            out_links_set = set()
             
            for a in soup('a', href=True):
                if a['href'] and a['href'][0] != '#':
                    link = urldefrag(urljoin(url, a['href'])).url
                    if link in valid_urls:
                        out_links_set.add(link)
                         
            out_degree = len(out_links_set)
            for l in out_links_set:
                g.add_edge(url, l, weight=1.0/out_degree)
             
        if docID % 100 == 0:
            print(docID)
    print("Done.")
         
    print('Computing PageRanks...')
    helper_PageRank(g, docDict)
    print("Done.")
        
# Host code
if __name__ == '__main__':
    # Initialize the data arrays
    A = numpy.full((3, 3), 0, numpy.float) # matrix containing all 3's
    A[0] = [.5, 0, .5]
    A[1] = [0, 0, .5]
    A[2] = [.5, 1, 0]
    B = numpy.full((3, 1), 1, numpy.float) # matrix containing all 4's
    
    # Copy the arrays to the device
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    
    # Allocate memory on the device for the result
    C_global_mem = cuda.device_array((3,1))
    
    # Configure the blocks
    threadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Start the kernel 
    matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
    
    # Copy the result back to the host
    C = C_global_mem.copy_to_host()
    
    print(C)
    
    print('test1:')
    B = C
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array((3,1))
    matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
    C = C_global_mem.copy_to_host()
    print(C)
     
    print('test2')
    D = C
    D_global_mem = cuda.to_device(D)
    C_global_mem = cuda.device_array((3,1))
    matadd[blockspergrid, threadsperblock](B_global_mem, D_global_mem, C_global_mem)
    C = C_global_mem.copy_to_host()
    print(C)
     
    print('test3')
    D = C
    D_global_mem = cuda.to_device(D)
    C_global_mem = cuda.device_array((3,1))
    scalarmul[blockspergrid, threadsperblock](D_global_mem, C_global_mem, 10.0)
    C = C_global_mem.copy_to_host()
    print(C)

    
