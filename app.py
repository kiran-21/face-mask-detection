# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:16:44 2020

@author: kiran
"""

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)



@app.route('/exec')
def parse1(name=None):
	import  detect_mask_video
	print("done")
	return render_template('index.html',name=name)

if __name__ == '__main__':
    app.run()
    app.debug = True