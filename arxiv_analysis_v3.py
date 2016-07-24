#!/usr/bin/env python

import numpy as np
import os
import pandas
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib
import re
import site

stop = stopwords.words('english')

if site.site == 'local':
    cpath = os.getcwd() + '/'
else:
    cpath = '/home/tilanukwatta/scicano/'
#dbpath = '/home/tilan/data/ext_data/arxiv/'
dbpath = cpath
df_file_name = "arxiv_papers.sqlite.db"

def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

def preprocessor(text):
    text = re.sub('\$.{1,}\$', '', text) # remove all the latex equations from the text
    text= re.sub('\&.{1,5}', '', text) # remove all non ascii characters from text
    text_list = [w for w in tokenizer_porter(text) if w not in stop]  # remove stop words
    output = ""
    for w in text_list:
        output = output + " " + w
    return output

def get_paper_info():
    conn = sqlite3.connect(dbpath + df_file_name)
    c = conn.cursor()
    #c.execute('SELECT * FROM arxiv_papers ORDER BY rowid')
    c.execute('SELECT * FROM arxiv_papers')
    conn.commit()
    results = c.fetchall()
    conn.close()
    df = pandas.DataFrame(results, columns=['url', 'title', 'authors', 'abstract'])
    return df

def xyplot(x, y, xmin, xmax, ymin, ymax, title, xlabel, ylabel, plot_name, line=1, y_log=0):
    plt.subplots_adjust(hspace=0.4)
    ax = plt.subplot(111)

    if y_log == 1:
        #ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')


    if (line == 1):
        ax.plot(x, y, 'bx-', linewidth=2)
        #ax.plot([0., 100.], [0., 100.], 'b', linewidth=1)
    else:
        ax.plot(x, y, 'ro')
        #ax.plot([0., 100.], [0., 100.], 'b', linewidth=1)

    if (xmin < xmax):
        ax.set_xlim(xmin, xmax)
    if (ymin < ymax):
        ax.set_ylim(ymin, ymax)

    #ax.axhline(linewidth=axis_width, color="k")
    #ax.axvline(linewidth=axis_width, color="k")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    #plt.show()
    plt.savefig(plot_name, bbox_inches='tight')
    plt.clf()

def perform_cluster_analysis(dataset):

    X = dataset

    print 'X Shape: ', X.shape

    K = range(1, 100)
    meandistortions = []
    cluster_centers = []
    for k in K:
        print k
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        #import ipdb; ipdb.set_trace() # debugging code
        #meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])
        meandistortions.append(kmeans.inertia_)
        cluster_centers.append(kmeans.cluster_centers_)
        #print 'k: ', k, ' Cluster Centers: ', kmeans.cluster_centers_


    plot_name = "elbow_plot.pdf"
    title = 'Selecting k with the Elbow Method'
    xlabel = 'k'
    ylabel = 'Average distortion'
    xyplot(K, meandistortions, 0, 0, 0, 0, title, xlabel, ylabel, plot_name, line=1, y_log=0)

def find_clusters(paper_text, num_clusters, search_text):

    k = num_clusters
    print 'Number of Clusters: ', k

    filename = "count_vectorizer.dat"
    model_filename = "kmeans_" + str(k) + ".dat"
    print filename, model_filename

    if os.path.exists(filename):
        vec = joblib.load(cpath + filename)
        count = vec[0]
        tfidf = vec[1]
    else:
        tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)

        text = []
        for i in range(len(paper_text)):
            #print 'Before: ', paper_text[k]
            #text.append(paper_text[i])
            text.append(preprocessor(paper_text[i]))
            #print 'After: ', titles[k]
            #import ipdb; ipdb.set_trace() # debugging code

        count = CountVectorizer()
        count.fit(text)
        bag = count.transform(text)
        tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
        tfidf.fit(bag)
        X = tfidf.transform(bag)

        vec = [count, tfidf]
        joblib.dump(vec, cpath + filename, compress=8)
        print 'X Shape: ', X.shape

    #print count.vocabulary_
    #perform_cluster_analysis(X)

    #print "bag shape: ", bag.shape
    #print "dataset shape: ", X.shape

    #import ipdb;ipdb.set_trace() # debugging code

    if os.path.exists(model_filename):
        model = joblib.load(cpath + model_filename)
        kmeans = model[0]
        predict = model[1]
    else:
        print "Clustering the data..."
        kmeans = KMeans(n_clusters=k, n_jobs=2)
        kmeans.fit(X)
        predict = kmeans.predict(X)
        model = [kmeans, predict]
        joblib.dump(model, cpath + model_filename, compress=8)
        print "Data  Clustering Completed."

    y = tfidf.transform(count.transform([preprocessor(search_text)]))

    #for i in range(10):
    #    print predict[i], paper_titles[i]

    y_cluster = kmeans.predict(y)
    print "Predicted cluster: ", y_cluster
    wh = np.where(predict == y_cluster)[0]

    #import ipdb;ipdb.set_trace() # debugging code

    target_papers = paper_titles[wh]
    for i in range(len(wh)):
        print wh[i], "  ", target_papers[i]

    #import ipdb;ipdb.set_trace() # debugging code
    return wh

def find_paper_idx(search_text, num_clusters):

    filename = "count_vectorizer.dat"
    if os.path.exists(cpath + filename):
        vec = joblib.load(cpath + filename)
        count = vec[0]
        tfidf = vec[1]

    filename = "kmeans_" + str(num_clusters) + ".dat"
    if os.path.exists(cpath + filename):
        model = joblib.load(cpath + filename)
        kmeans = model[0]
        predict = model[1]

    y = tfidf.transform(count.transform([preprocessor(search_text)]))

    y_cluster = kmeans.predict(y)

    #print "Predicted cluster: ", y_cluster
    wh = np.where(predict == y_cluster)[0]
    #import ipdb;ipdb.set_trace() # debugging code
    return wh

if __name__ == '__main__':

    papers = get_paper_info()
    #print papers

    #import ipdb; ipdb.set_trace() # debugging code

    paper_titles = papers['title'].values
    #paper_authors = papers['authors'].values
    paper_abstracts = papers['abstract'].values

    #paper_text = []
    #for k in range(len(paper_titles)):
    #    paper_text.append(paper_titles[k] + " " + paper_abstracts[k])

    paper_text = paper_titles

    #import ipdb;ipdb.set_trace() # debugging code

    search_text = "gamma-ray bursts"
    #search_text = "gamma-ray bursts are most powerful bursts in the universe"
    #search_text = "pbh"
    #search_text = "Radiation Transfer in Gamma-Ray Bursts"
    #search_text = "nova is a compact star that burst periodically"

    num_clusters = 500

    find_clusters(paper_text, num_clusters, search_text)

    #print find_paper_idx(search_text, num_clusters)[:50]

    #find_clusters(paper_text, 100, search_text)

    #import ipdb;ipdb.set_trace() # debugging code
