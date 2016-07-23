#!/usr/bin/env python

import csv
import numpy as np
import os
import pandas
import urllib2
from BeautifulSoup import BeautifulSoup
import sqlite3
import time

cpath = os.getcwd() + '/'
dbpath = '/home/tilan/data/ext_data/arxiv/'

def save_page(data, url):
    soup = BeautifulSoup(data)
    title = str(soup.text[soup.text.find("Title:")+6:soup.text.find("Authors:")]).replace('"', '')
    authors = str(soup.text[soup.text.find("Authors:")+8:soup.text.find("(Submitted")]).replace('"', '')
    abstract = str(soup.text[soup.text.find("Abstract:")+9:soup.text.find("CONTEXTComments:")]).replace('"', '')

    conn = sqlite3.connect(dbpath + "arxiv_papers.sqlite.db")
    c = conn.cursor()

    c.execute('CREATE TABLE IF NOT EXISTS arxiv_papers (url TEXT UNIQUE, title TEXT, authors TEXT, abstract TEXT)')

    rowStr = '"' + str(url) + '", "' + str(title) + '", "' + str(authors) + '", "' + str(abstract) + '" '

    #print 'INSERT INTO arxiv_papers VALUES (' + rowStr + ')'

    c.execute('INSERT INTO arxiv_papers VALUES (' + rowStr + ')')


    conn.commit()
    #c.execute('CREATE INDEX IF NOT EXISTS star_catalog_ra_decl_magV_idx ON star_catalog (ra, decl, magV);')
    conn.close()


def show_page(data, url):
    soup = BeautifulSoup(data)
    #print soup.title.string
    title = soup.text[soup.text.find("Title:")+6:soup.text.find("Authors:")]
    authors = soup.text[soup.text.find("Authors:")+8:soup.text.find("(Submitted")]
    abstract = soup.text[soup.text.find("Abstract:")+9:soup.text.find("CONTEXTComments:")]

    print url
    print title
    print authors
    print abstract
    print "\n"

    #import ipdb; ipdb.set_trace() # debugging code

    #for line in data.readlines():
    #    print line
        #l.append(line)
    #s = '\n'.join(l)
    #if s.find('radio') >= 0:
    #    count = count + 1
    #    print 'radio count: ', count, '  ', grb[0], '  ', url

def get_last_record():
    conn = sqlite3.connect(dbpath + "arxiv_papers.sqlite.db")
    c = conn.cursor()
    c.execute('SELECT url FROM arxiv_papers ORDER BY rowid')
    conn.commit()
    results = c.fetchall()
    conn.close()
    num = len(results)
    last_record = str(results[num-1][0]).split('/')[5]

    return last_record


if __name__ == '__main__':

    preURL = "https://arxiv.org/abs/astro-ph/"

    #year = ['94']
    #year = ['96']
    #year = ['92', '93', '94', '95', '96', '97', '98', '99', '00', '01', '02', '03', '04', '05', '06', '07']
    #year = ['92', '93', '94', '95', '96', '97', '98', '99']
    year = ['00', '01', '02', '03', '04', '05', '06', '07']
    #month = ['07', '08', '09', '10', '11', '12']
    month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    article = np.array(range(999)) + 1

    #last_record = get_last_record()  # this for articles on or before 1999
    last_record = '20' + get_last_record()  # this for articles on or after 2000
    #print last_record
    #import ipdb; ipdb.set_trace() # debugging code

    for yy in year:
        for mm in month:
            ar_end = 0
            for ar in article:
                url = preURL + yy + mm + str(ar).zfill(3)
                #url_id = url[-7:]  # this for articles on or before 1999
                url_id = '20' + url[-7:]  # this for articles on or after 2000

                #print "\n" + url + "\n"
                print last_record, url_id

                if long(url_id) > long(last_record):
                    if ar_end == 0:
                        try:
                            #print "\n" + url + "\n"
                            data = urllib2.urlopen(url)
                            #import ipdb; ipdb.set_trace() # debugging code
                        except:
                            ar_end = 1
                            #print "error: ", url
                            #import ipdb; ipdb.set_trace() # debugging code
                        time.sleep(3)
                        if ar_end == 0:
                            print "\n" + url + "\n"
                            save_page(data, url)
            #time.sleep(60)
        #time.sleep(120)

    """
    import ipdb; ipdb.set_trace() # debugging code
    #url = 'http://gcn.gsfc.nasa.gov/other/160104A.gcn3'

    count = 0
    for url, grb in zip(url_arr, grb_list):
        print 'Reading... ', url
        try:
            data = urllib2.urlopen(url)
        except:
            try:
                data = urllib2.urlopen(url2)
            except:
                print "error: ", url
        l = []
        s = ''
        for line in data.readlines():
            l.append(line)
        s = '\n'.join(l)
        if s.find('radio') >= 0:
            count = count + 1
            print 'radio count: ', count, '  ', grb[0], '  ', url

    print 'GRBs with radio observations: ', count
    """
    #import ipdb; ipdb.set_trace() # debugging code

