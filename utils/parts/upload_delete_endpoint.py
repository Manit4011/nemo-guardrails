import os
import json
import logging
import boto3
from botocore.config import Config
from pymongo import MongoClient
from bson import json_util
from utils.docDB import delete_documents_from_db
from utils.pdf_excel import process_pdf, process_excel_csv
from dotenv import load_dotenv

load_dotenv()
# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        record = event['Records'][0]
        eventName = record['eventName']
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        eventSource = record['eventSource']
        eventTime = record['eventTime']

        config = Config(connect_timeout=3600, read_timeout=3600, retries={'max_attempts': 2, 'mode': 'standard'})
        client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2", config=config)

        logger.info(f"Event received: {eventName} for file: {object_key}")

        if eventName == "ObjectCreated:Put" and bucket_name == "policybot":
            logger.info(f"Received a request to insert file({object_key}) into DB.")
            if object_key.endswith('.pdf'):
                process_pdf(client, bucket_name, object_key, eventSource, eventTime)
            elif object_key.endswith('.xlsx') or object_key.endswith('.csv'):
                process_excel_csv(client, bucket_name, object_key, eventSource, eventTime)
            logger.info("Lambda Function Completed successfully.")
            return {'statusCode': 200, 'body': json.dumps('Success!', default=json_util.default)}

        elif eventName == "ObjectRemoved:Delete" and bucket_name == "policybot":
            logger.info(f"Received a request to remove file({object_key}) into DB.")
            deleted_count = delete_documents_from_db(object_key)
            logger.info(f"Deleted {deleted_count} documents related to {object_key} from DocumentDB.")
            return {'statusCode': 200, 'body': json.dumps(f'Successfully deleted {deleted_count} documents related to {object_key} from DocumentDB.')}
        else:
            logger.info("No request received.")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps(f'Error: {str(e)}')}

# Test the lambda function
event = {
  "Records": [
    {
      "eventSource": "aws:s3",
      "eventTime": "2024-12-02T13:55:56.789Z",
      "eventName": "ObjectCreated:Put", #Removed:Delete", #
      "s3":{
        "bucket": {
            "name": "policybot",
        },
        "object": {
            "key": "Chat Bot - Feedbacks.xlsx",
        }
      }
    }
  ]
}
print(lambda_handler(event, None))
# import pymongo
# client = pymongo.MongoClient("mongodb://dbteam:Gemini123@docdb-2024-08-09-09-27-11.cluster-chw0weay2rkt.ap-south-1.docdb.amazonaws.com:27017/?tls=true&tlsCAFile=global-bundle.pem&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false")
# col = client["PolicyDB"]["Test"]
# cursor = col.find({'metadata.file_name': "Chat Bot - Feedbacks.xlsx"})
# data = list(cursor)

# if not data:
#     print("No records found.")
# else:
#     print(data)
# client.close()