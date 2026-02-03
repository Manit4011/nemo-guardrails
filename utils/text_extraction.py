import boto3
import time
import logging
from botocore.config import Config
from utils.bucket_call import upload_to_s3, save_text_to_s3

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_job(textract_client, s3_bucket_name: str, object_name: str) -> str:
    try:
        response = textract_client.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': s3_bucket_name, 'Name': object_name}}
        )
        job_id = response["JobId"]
        logger.info(f"Started Textract job with JobId: {job_id}")
        return job_id
    except Exception as e:
        logger.error(f"Failed to start Textract job: {e}")
        raise

def is_job_complete(textract_client, job_id: str) -> str:
    retries = 0
    max_retries = 120  # To avoid indefinite waiting
    sleep_time = 1

    while retries < max_retries:
        try:
            response = textract_client.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]
            logger.info(f"Job status: {status}")

            if status in ["SUCCEEDED", "FAILED"]:
                return status

            retries += 1
            # Increase sleep time progressively if necessary
            time.sleep(min(5, sleep_time))
            sleep_time += 1
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            raise

    raise TimeoutError(f"Textract job did not complete within expected time frame for JobId: {job_id}")

def get_job_results(textract_client, job_id: str) -> list:
    pages = []
    next_token = None

    while True:
        try:
            if next_token:
                response = textract_client.get_document_text_detection(JobId=job_id, NextToken=next_token)
            else:
                response = textract_client.get_document_text_detection(JobId=job_id)
            pages.append(response)
            logger.info(f"No. of pages received: {len(pages)}")

            next_token = response.get('NextToken')
            if not next_token:
                break
        except Exception as e:
            logger.error(f"Error retrieving job results: {e}")
            raise

    return pages

def extract_text_from_pdf(bucket: str, document_key: str) -> dict:
    try:
        config = Config(connect_timeout=3600, read_timeout=3600, retries={'max_attempts': 2, 'mode': 'standard'})
        textract_client = boto3.client(service_name='textract', region_name='ap-south-1')
        job_id = start_job(textract_client, bucket, document_key)
        status = is_job_complete(textract_client, job_id)

        if status == 'SUCCEEDED':
            response = get_job_results(textract_client, job_id)
            extracted_text = {}

            for res in response:
                for block in res.get('Blocks', []):
                    if block.get('BlockType') == 'LINE':
                        page_number = block.get('Page')
                        if page_number not in extracted_text:
                            extracted_text[page_number] = ''
                        extracted_text[page_number] += f"{block.get('Text', '')}\n"

            return extracted_text

        else:
            logger.error(f"Text detection failed with status: {status}")
            return None

    except Exception as e:
        logger.error(f"An error occurred during text extraction: {e}")
        return None