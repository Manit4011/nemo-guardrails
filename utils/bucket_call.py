import os
import boto3
import logging
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
load_dotenv()
s3_client = boto3.client(service_name='s3', region_name=os.getenv("REGION_NAME"))

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def generate_presigned_pdf_url(bucket_name, object_key, expiration=3600):
    """
    Generate a pre-signed URL to open a PDF in the browser's viewer.

    :param bucket_name: Name of the S3 bucket.
    :param object_key: Key of the PDF object in the bucket.
    :param expiration: Time in seconds for the URL to remain valid.
    :return: Pre-signed URL as a string.
    """
    try:
        s3_client = boto3.client('s3', region_name = os.getenv("REGION_NAME"))
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_key,
                'ResponseContentType': 'application/pdf',
            },
            ExpiresIn=expiration
        )
        return presigned_url
    
    except Exception as e:
        logger.error(f"Error generating presigned URL for {object_key}: {e}")
        return f"An error occurred: {e}"


def upload_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name) 
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"File {file_name} uploaded to {bucket}/{object_name}")
        return object_name
    except FileNotFoundError:
        print(f"The file {file_name} was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None
    except PartialCredentialsError:
        print("Incomplete credentials provided")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def upload_pdfs_in_folder(folder_path, bucket):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                upload_to_s3(file_path, bucket)


def delete_from_s3(bucket_name, object_key):
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=object_key)
        print(f"Deleted {object_key} from bucket {bucket_name}")
        return True
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except PartialCredentialsError:
        print("Incomplete credentials provided")
        return False
    except Exception as e:
        print(f"An error occurred while deleting from S3: {e}")
        return False
    
    
def save_text_to_s3(text, bucket, object_name):
    try:
        s3_client.put_object(Body=text, Bucket=bucket, Key=object_name)
        print(f"Text file saved to {bucket}/{object_name}")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def extract_file_from_s3(bucket, object_name):
    try:
        s3_response = s3_client.get_object(Bucket=bucket, Key=object_name)
        print(f"Object Retrived from {bucket}/{object_name}")
        return s3_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return exit(1)

# upload_to_s3(file_name="utils/files/Gemini-Crime Policy-2023-24.pdf", bucket="ddqbot")