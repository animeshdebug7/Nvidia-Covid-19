from flask import Flask, render_template, Response
import cv2
from keras.models import load_model
from Merged import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

zone = pd.read_csv('zone_data.csv')
zone['total'] = np.sum(zone, axis=1)

print("done with import")

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(mask_social(video = '4.mp4', show_frame=0, output='test_output_t.avi'), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/live')
def live():
    return Response(mask_social(video = 0, show_frame=0, output='test_output_t.avi'), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/info')
def info():
    zone = pd.read_csv('zone_data.csv')
    mask = pd.read_csv('mask_data.csv')

    plt.figure(figsize=(8,5)) ##00afff
    sns.set(rc = {'axes.facecolor': '00afff', 'axes.grid': True,})
    sns.lineplot(data = zone['green'], linewidth = 3
                ,label = 'Safe', color = 'green')
    sns.lineplot(data = zone['yellow'], linewidth = 4
                ,label = 'Not Safe', color = 'yellow')
    sns.lineplot(data = zone['red'], linewidth = 4
                ,label = 'Red Alert', color = 'red')
    plt.title('Social Distance Dashboard')
    plt.savefig('static/Social.png')

    plt.figure(figsize=(8,5)) ##00afff
    sns.set(rc = {'axes.facecolor': '00afff', 'axes.grid': True,})
    sns.lineplot(data = mask['Mask'], linewidth = 3
             ,label = 'With Mask', color = 'green')
    sns.lineplot(data = mask['No Mask'], linewidth = 4
             ,label = 'Without Mask', color = 'red')
    plt.title('Mask Detector Dashboard')
    plt.savefig('static/Mask.png')

    plt.figure(figsize=(8,5)) ##00afff
    sns.set(rc = {'axes.facecolor': '00afff', 'axes.grid': True,})
    sns.barplot(x = mask['Mask'] , data = mask)
    plt.title('Mask Bar Graph')
    plt.savefig('static/MaskBAr')

    plt.figure(figsize=(8,5)) ##00afff
    sns.set(rc = {'axes.facecolor': '00afff', 'axes.grid': True,})
    sns.lineplot(data = zone['green'] + zone['yellow'] + zone['red'], label ='Total Crowd')
    plt.title('Total People')
    plt.savefig('static/Total.png')

    return render_template('info.html')

if __name__=="__main__":
    app.run(debug=True)

