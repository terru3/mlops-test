import json

import boto3

def test_model():
    client = boto3.client('lambda')
    response = client.invoke(
        FunctionName='ml-model-development-ml_model',
        InvocationType='RequestResponse',
        Payload=json.dumps({"credit_score": 724,
                  "country": 'France',
                  "gender": "Female",
                  "age": 27,
                  "tenure": 2,
                  "balance": 87628.15,
                  "products_number": 3,
                  "credit_card": 1,
                  "active_member": 0,
                  "estimated_salary": 152513.96})
    )

    assert json.loads(response['Payload'].read())["churn"] == 0
    ## any need to decode utf-8 after read?