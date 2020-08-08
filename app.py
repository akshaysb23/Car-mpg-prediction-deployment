#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle


# In[2]:


app=Flask(__name__)
trial=pickle.load(open('trial.pkl','rb'))


# In[ ]:


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=trial.predict(final_features)
    output=round(prediction[0],2)
    return render_template('index.html',prediction_text='MPG of car is{}'.format(output))
if __name__=='__main__':
    app.run(debug=True)

