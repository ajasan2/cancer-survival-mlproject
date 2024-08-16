import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            age = float(request.form.get('age')),
            gender = request.form.get('gender'),
            tumor_type = request.form.get('tumor_type'),
            tumor_grade = request.form.get('tumor_grade'),
            tumor_location = request.form.get('tumor_location'),
            treatment = request.form.get('treatment'),
            treatment_outcome = request.form.get('treatment_outcome'),
            recurrence_time = float(request.form.get('recurrence_time')),
            recurrence_site = request.form.get('recurrence_site')
        )

        pred_df = data.get_data_as_df()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('index.html', results=results[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)