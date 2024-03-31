import logging
import os
import pickle

import boto3
import pandas as pd

from botocore.client import ClientError
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

BUCKET = 'churn-bucket-test'
MODEL_NAME = 'model.pkl'

########################################

def get_data():
    s3 = boto3.resource('s3')
    df = pd.read_csv(f"s3://{BUCKET}/bank_churn.csv")
    return df

def preprocess_data(df):
    """
    Preprocesses churn dataset, performs train-test split, label encodes categorical data and scales data.
    Uploads the encoders and scaler for downstream inference use, as well as categorical columns (hard-coded).
    """

    df = df.drop(columns=['customer_id'])
    X = df.drop(columns=['churn'])
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2, 
                                                        random_state = 2024)

    ## label encode categorical columns
    cat_cols = ["country", "gender", "credit_card", "active_member"]
    encoders = []
    for i in range(len(cat_cols)):
        enc = LabelEncoder()
        X_train[cat_cols[i]] = enc.fit_transform(X_train[cat_cols[i]])
        X_test[cat_cols[i]] = enc.transform(X_test[cat_cols[i]])
        encoders.append(enc)
    
    ## scale numerical columns
    scaler = StandardScaler()
    X_train.loc[:, ~X_train.columns.isin(cat_cols)] = scaler.fit_transform(X_train.loc[:, ~X_train.columns.isin(cat_cols)])
    X_test.loc[:, ~X_test.columns.isin(cat_cols)] = scaler.transform(X_test.loc[:, ~X_test.columns.isin(cat_cols)])

    for i in range(len(encoders)):
        upload_to_s3(encoders[i], BUCKET, f"{cat_cols[i]}_encoder.pkl")
    upload_to_s3(scaler, BUCKET, "scaler.pkl")
    upload_to_s3(cat_cols, BUCKET, "cat_cols.pkl")

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    """
    Initializes and trains logistic regression model, uploads to S3.
    """
    model = LogisticRegression(random_state=2024)
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print(f"Accuracy: {acc}")

    # upload trained model to S3
    upload_to_s3(model, BUCKET, MODEL_NAME)


def upload_to_s3(obj, bucket, file_name):
    """
    Uploads object to S3 bucket.
    This function is intended to be used for uploading a trained model, label encoders, or scalers.
    """
    s3 = boto3.resource('s3')

    # test if bucket already exists
    error_code = '200'
    try:
        s3.meta.client.head_bucket(Bucket=BUCKET)
    except ClientError as e:
        error_code = e.response['Error']['Code']

    # if not exist, create
    if error_code == '404':
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={'LocationConstraint': 'us-west-1'})
    pickle_byte_obj = pickle.dumps(obj)
    s3.Object(bucket, file_name).put(Body = pickle_byte_obj) ## add model to bucket


def make_preds(test_data):
    """
    Performs inference using trained logistic regression model, loaded from S3.
    `test_point` is a dictionary containing the relevant data variables and their values for the test instance(s).
    """
    s3 = boto3.resource('s3')

    # load model, encoders, scaler
    model = pickle.loads(s3.Object(BUCKET, MODEL_NAME).get()['Body'].read())
    cat_cols = pickle.loads(s3.Object(BUCKET, "cat_cols.pkl").get()['Body'].read())
    encoders = []
    for i in range(len(cat_cols)):
        enc = pickle.loads(s3.Object(BUCKET, f"{cat_cols[i]}_encoder.pkl").get()['Body'].read())
        encoders.append(enc)
    scaler = pickle.loads(s3.Object(BUCKET, "scaler.pkl").get()['Body'].read())

    print(f"{type(model).__name__} is ready for inference.")
    

    # if a single test instance:
    if not isinstance(list(test_data.values())[0], list):
        test_data = pd.DataFrame([test_data])
    else:
        test_data = pd.DataFrame.from_dict(test_data)
    
    # preprocess
    for enc, col in zip(encoders, cat_cols):
        test_data[col] = enc.transform(test_data[col])
    test_data.loc[:, ~test_data.columns.isin(cat_cols)] = scaler.transform(test_data.loc[:, ~test_data.columns.isin(cat_cols)])

    preds = model.predict(test_data)
    return preds


def train_handle(event, context):
    load_dotenv()
    df = get_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"Shapes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")
    train_model(X_train, y_train, X_test, y_test)
    # do I need to return some sort of response = {"statusCode": 200, "body": json.dumps({})}??

def infer_handle(event, context):
    if event.get("source") == "KEEP_LAMBDA_WARM":
        LOGGER.info("No ML work to do. Just staying warm...")
        return "Keeping Lambda warm"

    load_dotenv()
    return {
        "churn": make_preds(test_data=event)
    }

## https://github.com/AndreasMerentitis/SkLambdaDemo-logistic/tree/main 

# if __name__ == "__main__":
#     load_dotenv()
#     df = get_data()
#     X_train, X_test, y_train, y_test = preprocess_data(df)
#     print(f"Shapes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")
#     train_model(X_train, y_train, X_test, y_test)

#     test_data = {"credit_score": 724,
#                   "country": 'France',
#                   "gender": "Male",
#                   "age": 27,
#                   "tenure": 2,
#                   "balance": 87628.15,
#                   "products_number": 3,
#                   "credit_card": 1,
#                   "active_member": 0,
#                   "estimated_salary": 152513.96}
    
#     test_data = {"credit_score": [724, 516],
#                   "country": ['France', 'Germany'],
#                   "gender": ['Female', 'Male'],
#                   "age": [27, 51],
#                   "tenure": [2, 15],
#                   "balance": [87628.15, 5316426.41],
#                   "products_number": [3, 7],
#                   "credit_card": [1, 1],
#                   "active_member": [0, 1],
#                   "estimated_salary": [152513.96, 735136.31]}
    
#     preds = make_preds(test_data)
#     print(f"Final predictions: {preds}")

# Data Acknowledgements:
# https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

