import boto3
import json
import os

from botocore.exceptions import ClientError

def get_secret():
    secret_name = os.getenv("AWS_SECRET_NAME","policybot/prod")
    region_name = os.getenv("REGION_NAME","ap-south-1")
 
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
 
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)

    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
    secret = get_secret_value_response['SecretString']
    return json.loads(secret)

secrets = get_secret()
prompt_guardrail_arn=secrets['prompt_guardrail_arn']
DocDB_API_URL=secrets['DocDB_API_URL']
API_KEY=secrets['API_KEY']
FLASK_API_IP=secrets['FLASK_API_IP']
PROD_CHANNEL_ID=secrets['PROD_CHANNEL_ID']
PROD_CHANNEL_ID2=secrets['PROD_CHANNEL_ID2']
ITSM_KEY = secrets['ITSM_KEY']
ITSM_CHANNEL_ID = secrets['ITSM_CHANNEL_ID']