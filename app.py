from __future__ import division, print_function
from scipy import fftpack
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import keras
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from werkzeug.utils import secure_filename
from datetime import datetime
from keras.preprocessing import image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template,send_file
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import Flask, render_template, request
from flask import Response
from matplotlib.figure import Figure
import pandas as pd

x = [i for i in range(200)]

a = pd.read_csv('web.csv')
def plott(df):
    af3 = df['EEG.AF3'] 
    yaf3 = af3.head(200)

    f7 = df['EEG.F7']
    yf7 = f7.head(200)

    f3 = df['EEG.F3']
    yf3 = f3.head(200)

    fc5 = df['EEG.FC5'] 
    yfc5 = fc5.head(200)

    t7 = df['EEG.T7']
    yt7 = t7.head(200)

    p7 = df['EEG.P7'] 
    yp7 = p7.head(200)

    o1 = df['EEG.O1']
    yo1 = o1.head(200)

    o2 = df['EEG.O2'] 
    yo2 = o2.head(200)

    p8 = df['EEG.P8']
    yp8 = p8.head(200)

    t8 = df['EEG.T8'] 
    yt8 = t8.head(200)

    fc6 = df['EEG.FC6']
    yfc6 = fc6.head(200)

    f8 = df['EEG.F8'] 
    yf8 = f8.head(200)

    f4 = df['EEG.F4'] 
    yf4 = f4.head(200)

    af4 = df['EEG.AF4'] 
    yaf4 = af4.head(200)

    plt.subplot(7,2,1)
    plt.plot(x,yaf3)
    plt.subplot(7,2,2)
    plt.plot(x,yf7)
    plt.subplot(7,2,3)
    plt.plot(x,yf3)
    plt.subplot(7,2,4)
    plt.plot(x,yfc5)
    plt.subplot(7,2,5)
    plt.plot(x,yt7)
    plt.subplot(7,2,6)
    plt.plot(x,yp7)
    plt.subplot(7,2,7)
    plt.plot(x,yo1)
    plt.subplot(7,2,8)
    plt.plot(x,yo2)
    plt.subplot(7,2,9)
    plt.plot(x,yp8)
    plt.subplot(7,2,10)
    plt.plot(x,yt8)
    plt.subplot(7,2,11)
    plt.plot(x,yfc6)
    plt.subplot(7,2,12)
    plt.plot(x,yf4)
    plt.subplot(7,2,13)
    plt.plot(x,yf8)
    plt.subplot(7,2,14)
    plt.plot(x,yaf4)
    plt.savefig("static\pics\inal.jpg")
    plt.show()
def noiseRemove(df1):
    time_step = 0.05
    time_vec = np.arange(0,10,time_step)
    af3 = df1['EEG.AF3'] 
    yaf3 = af3.head(200)

    f7 = df1['EEG.F7']
    yf7 = f7.head(200)

    f3 = df1['EEG.F3']
    yf3 = f3.head(200)

    fc5 = df1['EEG.FC5'] 
    yfc5 = fc5.head(200)

    t7 = df1['EEG.T7']
    yt7 = t7.head(200)

    p7 = df1['EEG.P7'] 
    yp7 = p7.head(200)

    o1 = df1['EEG.O1']
    yo1 = o1.head(200)

    o2 = df1['EEG.O2'] 
    yo2 = o2.head(200)

    p8 = df1['EEG.P8']
    yp8 = p8.head(200)

    t8 = df1['EEG.T8'] 
    yt8 = t8.head(200)

    fc6 = df1['EEG.FC6']
    yfc6 = fc6.head(200)

    f8 = df1['EEG.F8'] 
    yf8 = f8.head(200)

    f4 = df1['EEG.F4'] 
    yf4 = f4.head(200)

    af4 = df1['EEG.AF4'] 
    yaf4 = af4.head(200)
    
    sig = np.array(yaf3)
    np.round(sig,2)
    sig_fft = fftpack.fft(sig)
    Amlitude = np.abs(sig_fft)
    Power = Amlitude**2
    Angle = np.angle(sig_fft)
    sample_freq = fftpack.fftfreq(sig.size,d=time_step)
    Amp_Freq = np.array([Amlitude, sample_freq])
    Amp_position = Amp_Freq[0,:].argmax()
    peak_freq = Amp_Freq [1, Amp_position]
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq)>5.0]=0
    high_freq_fft[np.abs(sample_freq)==4.0]
    filtered_sig = fftpack.ifft(high_freq_fft)
    plt.subplot(2,1,1)
    plt.plot(time_vec,sig)
    plt.subplot(2,1,2)
    plt.plot(time_vec,filtered_sig)
   
    plt.savefig('static\pics\oise1')
    plt.subplot(2,1,1)
    plt.plot(time_vec,sig)
    plt.subplot(2,1,2)
    plt.plot(x,x)
    plt.savefig('static\pics\oise2')



ALLOWED_EXTENTIONS = set(['csv'])
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENTIONS




app = Flask(__name__)
MODEL_PATH = 'D:\EEG project\RunModel'
model = load_model(MODEL_PATH)
picFolder = os.path.join('static','pics')
app.config['UPLOAD_FOLDER'] = picFolder
@app.route('/')
def index(): #home = index
    # pic1 = os.path.join(app.config['UPLOAD_FOLDER'],'outindex.jpg')
    return render_template("index.html")

@app.route('/upload', methods=['GET','POST'])
def upolad():
    if request.method == 'POST':
        file = request.files['file']
        file.save('test.csv')
        # if file and allowed_file(file.filename):
        #     filename= secure_filename(file.filename)
        #     new_filename = f'{filename.split(".")[0]}_{str(datetime.now())}.jpg'
        #     file.save(os.path.join(('input',new_filename)))

        #return redirect(url_for('download'))
        df1 = pd.read_csv('test.csv')
        # AF3 = df1['EEG.AF3']
        # yy = AF3.head(200)
        # xx = [i for i in range(200)]
        # plt.plot(xx,yy)
        # plt.savefig("static\pics\webindex.jpg")
        plott(df1)
        pic2 = os.path.join(app.config['UPLOAD_FOLDER'],'inal.jpg')
        return render_template("index.html", user_name=pic2 , user_name1=pic2 )
# df = pd.read_csv('test.csv')
# AF3 = df['EEG.AF3']
# yy = AF3.head(200)
# xx = [i for i in range(200)]
# plt.plot(xx,yy)
# plt.savefig("static\pics\outindex.jpg")

# def index(): #home = index
#     # csv = request.files['csv']
#     # df = pd.read_csv(csv)
#     # AF3 = df['EEG.AF3']
#     # yy = AF3.head(200)
#     # xx = [i for i in range(200)]
#     # plt.plot(xx,yy)
#     # plt.savefig("static\pics\outindex.jpg")
#     pic1 = os.path.join(app.config['UPLOAD_FOLDER'],'outindex.jpg')
#    return render_template("index.html")
# @app.route("/output", methods=["POST"])
# def output():
#     df = pd.read_csv('test.csv')
#     AF3 = df['EEG.AF3']
#     yy = AF3.head(200)
#     xx = [i for i in range(200)]
#     plt.plot(xx,yy)
#     plt.savefig("static\pics\outindex1.jpg")
#     pic2 = os.path.join(app.config['UPLOAD_FOLDER'],'outindex1.jpg')
#     return render_template("output.html",user_image1 = pic2)

@app.route("/prediction", methods=["POST"])
def prediction():
    img = request.files['img']
    print(img)
    img.save('img.jpg')
    img_path = 'img.jpg'
    img = keras.preprocessing.image.load_img(img_path, target_size=(150,150))
    x = keras.preprocessing.image.img_to_array(img)
    x = tf.expand_dims(x, 0) 
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    if preds==0:
        y = 'Understood'
    else:
        y = 'Not_Understood'
    return render_template("prediction.html",data = y)

if __name__ == '__main__':
    app.run(debug=True)