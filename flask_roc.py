from flask import Flask
from flask_restful import Resource, Api
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)

class ROC(Resource):
    def get(self, preprocessing, c):
		 # you need to preprocess the data according to user preferences (only fit preprocessing on train data)
        if(preprocessing=="standard"):
            scaler = StandardScaler(with_std=False)
            standardized_train=scaler.fit_transform(X_train)
            standardized_test=scaler.transform(X_test)
        if(preprocessing=="minmax"):
            scaler=MinMaxScaler()
            standardized_train=scaler.fit_transform(X_train)
            standardized_test=scaler.transform(X_test)
		# fit the model on the training set
        lr=LogisticRegression(C=c)
        lr.fit(standardized_train,y_train)
		# predict probabilities on test set
        score = lr.predict_proba(standardized_test)
        fpr, tpr, threshold = roc_curve(y_test, score[:, 1], pos_label = 1)
        json_list = []
        for fpr, tpr in zip(fpr, tpr):
            json_list.append({"fpr":fpr, "tpr":tpr})
        return json_list
# Here you need to add the ROC resource, ex: api.add_resource(HelloWorld, '/')
api.add_resource(ROC, '/<string:preprocessing>/<float:c>')
# for examples see 
# https://flask-restful.readthedocs.io/en/latest/quickstart.html#a-minimal-api

if __name__ == '__main__':
    # load data
    df = pd.read_csv('data/transfusion.data')
    xDf = df.loc[:, df.columns != 'Donated']
    y = df['Donated']
	# get random numbers to split into train and test
    np.random.seed(1)
    r = np.random.rand(len(df))
	# split into train test
    X_train = xDf[r < 0.8]
    X_test = xDf[r >= 0.8]
    y_train = y[r < 0.8]
    y_test = y[r >= 0.8]
    app.run(debug=True)
