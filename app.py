
from flask import Flask, render_template, request
from flask import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__,template_folder="templates")
model2 = pickle.load(open('model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Visibility=0.0539
    if request.method == 'POST':
        Fat = float(request.form['Item_Fat_Content'])
        Item_type = float(request.form['Item_Type'])
        Location=float(request.form['Outlet_Location_Type'])
        Outlet_type = float(request.form['Outlet_Type'])
        Age = float(request.form['Age_Outlet'])
        Price = float(request.form['Item_MRP'])      
        prediction=model2.predict([[Fat,Visibility,Item_type,Price,Location,Outlet_type,Age]])
        output=prediction[0]
        print(output)
        output="{:.2f}".format(output)
        if output==0:
            return render_template('index.html',prediction_text="Sale cannot be predicted due to invalid entries")
        else:
            return render_template('index.html',prediction_text=output)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)