import json

import boto3

# from dotenv import load_dotenv
# load_dotenv()

def test_model():
    client = boto3.client('lambda') ## region temp for local pytesting
    # client = boto3.client('lambda', region_name='us-west-1') ## region temp for local pytesting
    response = client.invoke(
        FunctionName='ml-model-development-ml_model',
        InvocationType='RequestResponse',
        Payload=json.dumps({"credit_score": 724,
                  "country": "France",
                  "gender": "Female",
                  "age": 27,
                  "tenure": 2,
                  "balance": 87628.15,
                  "products_number": 3,
                  "credit_card": 1,
                  "active_member": 0,
                  "estimated_salary": 152513.96})
    )
    assert json.loads(response['Payload'].read().decode('utf-8'))["churn"][0] == 0