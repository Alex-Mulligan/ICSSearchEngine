'''
Created on Feb 27, 2020

@author: ajm00
'''

import time
from flask import Flask, render_template, request, redirect
from engine import Engine
from flask.helpers import url_for
from nltk.stem import PorterStemmer

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html.j2")

@app.route("/create", methods=['POST'])
def create():
    f = request.form
    print(f)
    query = f['query']
    if not query:
        return redirect(url_for('home'))
    return redirect(url_for('results', query=query))

@app.route("/search/<query>")
def results(query):
    start = time.time()
    ps = PorterStemmer()
    terms = [ps.stem(w) for w in query.split()]
    options = e.rank_search(terms)
    sites = [(e.url_dict[str(options[i][0])], e.titles[str(options[i][0])] if str(options[i][0]) in e.titles.keys() else "") for i in range(0, min(len(options), 10))]
#     sites = [('testsite1',7),('testsite2',4),('testsite3',2),('testsite4',2),('testsite5',1)]
    query_time = time.time() - start
    num_sites = len(options)
#     titles = [e.titles[d[0]] for d in sites]
    return render_template('results.html.j2', **locals())

if __name__ == '__main__':
    global e
    e = Engine()
    app.run()
    