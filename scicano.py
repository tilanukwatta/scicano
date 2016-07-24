from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators, StringField
import numpy as np
import sqlite3
import os
import pandas
import arxiv_analysis_v3 as model
import scicano_site

app = Flask(__name__)

if scicano_site.site == 'local':
    cpath = os.getcwd() + '/'
else:
    cpath = '/home/tilanukwatta/scicano/'
#dbpath = '/home/tilan/data/ext_data/arxiv/'
dbpath = cpath
df_file_name = "arxiv_papers.sqlite.db"

class searchForm(Form):
    #search_text = TextAreaField('', [validators.DataRequired()])
    search_text = StringField('', [validators.DataRequired()])

def get_paper_info(index):
    conn = sqlite3.connect(dbpath + df_file_name)
    c = conn.cursor()

    #c.execute('SELECT * FROM arxiv_papers ORDER BY rowid')
    #import ipdb; ipdb.set_trace() # debugging code
    c.execute('SELECT * FROM arxiv_papers WHERE rowid in ({0})'.format(', '.join('?' for _ in index)), index)

    conn.commit()
    results = c.fetchall()
    conn.close()
    df = pandas.DataFrame(results, columns=['url', 'title', 'authors', 'abstract'])
    return df

@app.route('/')
def index():
    form = searchForm(request.form)
    return render_template('home.html', form=form)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/results', methods=['POST'])
def results():
    form = searchForm(request.form)
    if request.method == 'POST' and form.validate():
        search_text = request.form['search_text']

        #index = np.random.randint(1, 1000, 10)
        #index = model.find_paper_idx(search_text, 500)[:25]+1
        index = model.find_paper_idx(search_text, 500)+1
        #import ipdb; ipdb.set_trace() # debugging code

        results = get_paper_info(index).values

        return render_template('home.html', results=results, form=form)

    return render_template('home.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)

    #df = get_paper_info([34, 56, 899, 23, 332])
    #df = get_paper_info(('34', ))
    #print df
    #import ipdb; ipdb.set_trace() # debugging code

