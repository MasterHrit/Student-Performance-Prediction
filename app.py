import pickle
from flask import Flask,request,render_template,redirect,url_for
import numpy as np
import pandas as pd
import math

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application=Flask(__name__)

app=application

@app.route("/")
def index():
    return redirect(url_for("predict_datapoint"))

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if(request.method=="GET"):
        return render_template("home.html")
    else:
        data=CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=request.form.get("reading_score"),
            writing_score=request.form.get("writing_score")
        )
        df=data.get_data_as_frame()
        pred_obj=PredictPipeline()
        result=pred_obj.predict(df)
        return render_template("home.html",results=np.round(result[0],2))

app.run(debug=True)