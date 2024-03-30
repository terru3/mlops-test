import boto3
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def get_data():
    df = pd.read_csv("bank_churn.csv")

    ## TODO: preprocess data
    ## then maybe upload preprocessed to S3?
    df = df.drop(columns=['customer_id'])
    X = df.drop(columns=['churn'])
    y = df['churn']

    X["country"] = LabelEncoder().fit_transform(X["country"])
    X["gender"] = LabelEncoder().fit_transform(X["gender"])

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2, 
                                                        random_state = 2024)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(random_state=2024)
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print(f"Accuracy: {acc}")
    # ## upload trained model to S3?
    # s3 = boto3.resource('s3')
    # bucket = 'your_bucket'
    # key = 'pickle_model.pkl'
    # pickle_byte_obj = pickle.dumps(model)
    # s3.Object(bucket, key).put(Body = pickle_byte_obj) ## add object to bucket



## then for inference:

## download model, preprocess data

# s3 = boto3.resource('s3')
# bucket = 'your_bucket'
# key = 'pickle_model.pkl'
# my_pickle = pickle.loads(s3.Object(bucket, key).get()['Body'].read())


## preds = model.predict()
# response = {
#             "statusCode": 200,
#            "body": json.dumps(preds)} # or smth
## return response



## in serverless, have train function, infer function, etc.

## https://github.com/AndreasMerentitis/SkLambdaDemo-logistic/tree/main 

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    print(f"Shapes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")
    train_model(X_train, y_train, X_test, y_test)
