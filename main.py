# -*- coding: utf-8 -*-

import os
from flask import Flask, render_template, request, redirect, url_for

from imglibrary import ImgLibrary, IMG_LIBRARY_DIR


imglib = ImgLibrary()
imglib.reload()
imglib.load_all()
imglib.sort_by_alphabet()


app = Flask(__name__)


@app.route('/')
def index():
    if request.args.get("sort") == "alphabet":
        imglib.sort_by_alphabet()
        imglib.inverse = False
    elif request.args.get("sort") == "alphabet_alt":
        imglib.sort_by_alphabet_alt()
        imglib.inverse = False
    elif request.args.get("sort") == "inv":
        imglib.inverse = not imglib.inverse
    else:
        redirect(url_for('index'))
    
    images = imglib()
    return render_template('index.html', images=images,
                           libdir=os.path.relpath(IMG_LIBRARY_DIR),
                           sentence=imglib.sentence)



@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        sentence = request.form['search']
        if sentence:
            imglib.search(sentence)
            imglib.sentence = sentence
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
    app.run()
