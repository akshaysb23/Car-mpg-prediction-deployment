#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle


# In[5]:


app=Flask(__name__)
trial=pickle.load(open('deployment.pkl','rb'))


# In[6]:


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


# In[ ]:




