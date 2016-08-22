#!/usr/bin/env python

""" Implements the recommendation engine of the scicano web application

Version 1.0

Purpose
=======

The purpose of this program is to implement the recommendation engine
of scicano. It also has routines to make diagnostic plots visualize
various aspects of the engine.

"""

import numpy as np
import os
import pandas
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib
import re
import scicano_site

stop = stopwords.words('english')

if scicano_site.site == 'local':
    cpath = os.getcwd() + '/'
    staticpath = os.getcwd() + '/static/'
    #dbpath = cpath
    dbpath = '/home/tilan/data/ext_data/arxiv/'
else:
    cpath = '/home/tilanukwatta/scicano/'
    staticpath = '/home/tilanukwatta/scicano/static/'
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
        ax.plot(x, y, 'b', linewidth=2)
        ax.plot(x, y, 'ro')
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

def xybarplot(x, y, xmin, xmax, ymin, ymax, title, xlabel, ylabel, plot_name):
    plt.subplots_adjust(hspace=0.4)
    ax = plt.subplot(111)

    ax.bar(x, y, 1.0, color='blue', align='center')

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

    filename = 'elbow_plot.dat'

    if os.path.exists(cpath + filename):
        data = joblib.load(cpath + filename)
        K = data[0]
        meandistortions = data[1]
    else:
        X = dataset
        print 'X Shape: ', X.shape

        #K = range(1, 50, 5)
        K = [1, 2, 5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        #K = [1, 2, 5, 10, 50, 100]
        meandistortions = []
        cluster_centers = []
        for k in K:
            print k
            kmeans = KMeans(n_clusters=k, n_jobs=3)
            kmeans.fit(X)
            #import ipdb; ipdb.set_trace() # debugging code
            #meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])
            meandistortions.append(kmeans.inertia_)
            cluster_centers.append(kmeans.cluster_centers_)
            #print 'k: ', k, ' Cluster Centers: ', kmeans.cluster_centers_
        data = [K, meandistortions]
        joblib.dump(data, cpath + filename, compress=8)

    plot_name = "elbow_plot.png"
    title = 'Selecting k with the Elbow Method'
    xlabel = 'Number of Clusters (k)'
    ylabel = 'Average Distortion'
    xyplot(K, meandistortions, 0, 0, 0, 0, title, xlabel, ylabel, staticpath + plot_name, line=1, y_log=0)

def plot_cluster_number_distribution(predictions, num_clusters):

    filename = 'cluster_num_dist_plot.dat'

    if os.path.exists(cpath + filename):
        data = joblib.load(cpath + filename)
        x = data[0]
        y = data[1]
    else:
        x = []
        y = []
        for i in range(num_clusters):
            wh = np.where(predictions == i)[0]
            x.append(i)
            y.append(len(wh))
        data = [x, y]
        joblib.dump(data, cpath + filename, compress=8)

    plot_name = "cluster_num_dist_plot.png"
    title = 'Cluster Member Distribution'
    xlabel = 'Cluster Label'
    ylabel = 'Number of Publications'
    #import ipdb;ipdb.set_trace() # debugging code
    xybarplot(x, y, 0, num_clusters, 0, 0, title, xlabel, ylabel, staticpath + plot_name)

def find_clusters(paper_text, num_clusters, search_text):

    k = num_clusters
    print 'Number of Clusters: ', k

    filename = "count_vectorizer_new.dat"
    model_filename = "kmeans_" + str(k) + "_new.dat"
    print filename, model_filename

    if os.path.exists(cpath + filename):
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

        # plot elbow plot to look at the effect of different number of clusters
        print "Perform cluster analysis..."
        perform_cluster_analysis(X)
        print "Perform cluster analysis...done."

    #print count.vocabulary_
    #perform_cluster_analysis(X)

    #print "bag shape: ", bag.shape
    #print "dataset shape: ", X.shape

    #import ipdb;ipdb.set_trace() # debugging code

    if os.path.exists(cpath + model_filename):
        model = joblib.load(cpath + model_filename)
        kmeans = model[0]
        predict = model[1]
    else:
        print "Clustering the data..."
        kmeans = KMeans(n_clusters=k, n_jobs=3)
        kmeans.fit(X)
        predict = kmeans.predict(X)
        model = [kmeans, predict]
        joblib.dump(model, cpath + model_filename, compress=8)
        print "Data  Clustering Completed."

    #for i in range(10):
    #    print predict[i], paper_titles[i]
    print "Plotting the cluster distribution..."
    plot_cluster_number_distribution(predict, num_clusters)
    print "Plotting the cluster distribution...done."
    #import ipdb;ipdb.set_trace() # debugging code

    """
    y = tfidf.transform(count.transform([preprocessor(search_text)]))

    y_cluster = kmeans.predict(y)
    #print "Predicted cluster: ", y_cluster
    wh = np.where(predict == y_cluster)[0]
    #import ipdb;ipdb.set_trace() # debugging code

    target_papers = paper_titles[wh]
    for i in range(len(wh)):
        print wh[i], "  ", target_papers[i]

    #import ipdb;ipdb.set_trace() # debugging code
    return wh
    """

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
    #num_clusters = 1000

    find_clusters(paper_text, num_clusters, search_text)

    #print find_paper_idx(search_text, num_clusters)[:50]

    #find_clusters(paper_text, 100, search_text)

    #import ipdb;ipdb.set_trace() # debugging code
