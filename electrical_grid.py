import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sys
import os
import re
import sklearn
import joblib
from joblib import load
from flask import Flask, render_template, url_for, flash, redirect, request, send_from_directory
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow.keras.models import load_model

app = Flask(__name__)

model_grid = load_model('Electrical_Grid_Stability.h5')
scaler=load('Scaler.joblib')

@app.route('/',methods=['GET'])

@app.route('/electrical_grid', methods=['GET','POST'])
def electrical_grid():
    if request.method == 'POST':
        tau1 = float(request.form['tau1'])
        tau2 = float(request.form['tau2'])
        tau3 = float(request.form['tau3'])
        tau4 = float(request.form['tau4'])
        p1 = float(request.form['p1'])
        p2 = float(request.form['p2'])
        p3 = float(request.form['p3'])
        p4 = float(request.form['p4'])
        g1 = float(request.form['g1'])
        g2 = float(request.form['g2'])
        g3 = float(request.form['g3'])
        g4 = float(request.form['g4'])
        
        X_test = scaler.transform([[tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4]])
        prediction = model_grid.predict(X_test)
        
        if prediction[0][0] >= 0:
            return render_template('electrical_grid.html', prediction_text="Oops! the system is linearly unstable with a stability value of {:.5f}.".format(prediction[0][0]), title='Electrical Grid Stability')
        else:
            return render_template('electrical_grid.html', prediction_text="Great! the system is stable with a stability value of {:.5f}.".format(prediction[0][0]), title='Electrical Grid Stability')
    else:
        return render_template('electrical_grid.html', title='Electrical Grid Stability')


if __name__=='__main__':
    app.run(debug=True)

