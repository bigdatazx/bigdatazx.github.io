# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:30:48 2019

@author: Administrator
"""
from flask import Flask, request,render_template
from keras.preprocessing.image import ImageDataGenerator,image
from werkzeug.utils import secure_filename # 使用这个是为了确保filename是安全的
from os import path
import os
from flask_cors import *
import json
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
print("load model...")
global modelPlant
modelPlant = load_model('new_gpu_plantvillage_best_model7.hdf5')
print("load done")
app = Flask(__name__)
CORS(app)#跨域

ALLOWED_EXIENSIONS = set(['txt','pdf','png','jpg','jpeg','gif'])
#=========================================================备用===================================
def decode_predictions_ill(preds):#输出格式
    CLASS_ILL = ["c_0","c_1","c_10","c_11","c_12","c_13","c_14","c_15","c_16","c_17","c_18",
                "c_19","c_2","c_20","c_21","c_22","c_23","c_24","c_25",
                "c_26","c_27","c_28","c_29","c_3","c_30","c_31","c_32",
                "c_33","c_34","c_35","c_36","c_37","c_4","c_5","c_6",
                "c_7","c_8","c_9"]
    results = {}
    top_indices = np.argmax(preds)
    print(CLASS_ILL)
    print(top_indices)
    print(preds.shape)
    results[CLASS_ILL[top_indices]] = str(preds[0][top_indices]*100) 
    print("predict end...")
    return results

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXIENSIONS#检查拓展名是否合法

@app.route('/',methods=['GET','POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			return  redirect(request.url)
		file = request.files['file']
		if file.filename =='':
			return redirect(request.url)
		if file and allowed_file(file.filename):
			base_path = path.abspath(path.dirname(__file__))
			upload_path = path.join(base_path,'static/planttest/test/')
			filename = upload_path+secure_filename(file.filename)
			file.save(filename)
			try:
				test_datagen = ImageDataGenerator(rescale=1./255)
				img = image.load_img(file,target_size=(224,224))
				array_img = image.img_to_array(img,dtype='float32')
				test_generator = test_datagen.flow(array_img.reshape(1,224,224,3),batch_size=1)
				pred = modelPlant.predict_generator(test_generator)
				result = decode_predictions_ill(pred)
				return json.dumps(result)
			except:
				result = {'error':'predict error...'}
				return json.dumps(result)
	return render_template('index.html')
if __name__ == '__main__':
	app.run()
