
from flask import Flask, render_template, Response, request, send_file, jsonify, redirect, url_for
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import glob 
import json 

from utils.query_processing import Translation
from utils.faiss import Myfaiss

# http://0.0.0.0:5001/home?index=0

# app = Flask(__name__, template_folder='templates', static_folder='static')

app = Flask(__name__, template_folder='templates')

####### CONFIG #########
with open('image_path.json') as json_file:
    json_dict = json.load(json_file)

DictImagePath = {}
for key, value in json_dict.items():
   DictImagePath[int(key)] = value 

LenDictPath = len(DictImagePath)
bin_file='faiss_normal_ViT.bin'
MyFaiss = Myfaiss(bin_file, DictImagePath, 'cpu', Translation(), "ViT-B/32")
########################

# Example definitions for testing purposes (replace with your actual data)
# Example definitions (replace these with your actual data)
LenDictPath = 1000  # Total number of images
DictImagePath = [f'path/to/image/{i}.jpg' for i in range(LenDictPath)]  # Sample image paths

@app.route('/home')
@app.route('/')
def thumbnailimg():
    print("load_iddoc")
    
    # Get the index from the request, default to 0
    index_str = request.args.get('index', default='0')

    try:
        # Convert the index to an integer
        index = int(index_str)
        return redirect(url_for('text_search', textquery=''))

    except ValueError:
        return "Index must be an integer", 400

    imgperindex = 100  # Number of images per index
    pagefile = []

    # Calculate first and last index for slicing
    first_index = index * imgperindex
    last_index = min(first_index + imgperindex, LenDictPath)

    # Ensure first_index is within bounds
    if first_index < LenDictPath:
        page_filelist = DictImagePath[first_index:last_index]
        list_idx = range(first_index, last_index)

        # Populate pagefile with image paths and their IDs
        for imgpath, id in zip(page_filelist, list_idx):
            pagefile.append({'imgpath': imgpath, 'id': id})

    # Prepare data for rendering
    data = {
        'num_page': (LenDictPath // imgperindex) + 1,
        'pagefile': pagefile
    }

    return render_template('home.html', data=data)


@app.route('/imgsearch')
def image_search():
    print("image search")
    pagefile = []
    id_query = int(request.args.get('imgid'))
    _, list_ids, _, list_image_paths = MyFaiss.image_search(id_query, k=50)

    imgperindex = 100 

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('home.html', data=data)

@app.route('/textsearch')
def text_search():
    print("text search")

    pagefile = []
    text_query = request.args.get('textquery')
    if(text_query == ""):
        datas = "langdetect.lang_detect_exception.LangDetectException: No features in text."
        return render_template('home.html', data=datas)
    else:
        _, list_ids, _, list_image_paths = MyFaiss.text_search(text_query, k=50)

        imgperindex = 100 

        for imgpath, id in zip(list_image_paths, list_ids):
            pagefile.append({'imgpath': imgpath, 'id': int(id)})

        data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
        
        return render_template('home.html', data=data)

@app.route('/get_img')
def get_img():
    # print("get_img")
    fpath = request.args.get('fpath')
    # fpath = fpath
    list_image_name = fpath.split("/")
    image_name = "/".join(list_image_name[-2:])

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jph")
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.resize(img, (1280,720))

    # print(img.shape)
    img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   3, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
