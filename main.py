# -*- coding: utf-8 -*-

import os
from flask import Flask, render_template, request, redirect, url_for, session

from imglibrary import ImgLibrary, IMG_LIBRARY_DIR


imglib = ImgLibrary()
imglib.reload()


app = Flask(__name__)
app.secret_key = 'pixgallery'


@app.route('/')
def index():
    if 'sentence' not in session:
        session['sentence'] = ''
    
    if 'indexs' not in session:
        session['indexs'] = imglib.argsort_by_alphabet()
    
    if request.args.get("sort") == "alphabet":
        session['indexs'] = imglib.argsort_by_alphabet()
    elif request.args.get("sort") == "alphabet_alt":
        session['indexs'] = imglib.argsort_by_alphabet_alt()
    elif request.args.get("sort") == "inv":
        session['indexs'] = imglib.inverse(session['indexs'])
    
    images = imglib.sort(session['indexs'])
    
    return render_template('index.html', images=images,
                           libdir=os.path.relpath(IMG_LIBRARY_DIR),
                           sentence=session['sentence'])


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        session['sentence'] = request.form['search']
        if session['sentence']:
            session['indexs'] = imglib.argsearch(session['sentence'])
    return redirect(url_for('index'))


# static url cache buster ----
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)
# ---- static url cache buster


if __name__ == '__main__':
    app.run(port=3000)
