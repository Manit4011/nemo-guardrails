from flask import Flask, request, jsonify, Response
import json
import boto3
import logging
import time
from secrets_manager import API_KEY
from dotenv import load_dotenv
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from utils.docDB import get_past_queries, store_query, store_response, similarity_search
from bedrock_call import get_prompt, get_rewriting_prompt

load_dotenv()
# Initialize Flask app
app = Flask(__name__)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

app.config['CORS_HEADERS'] = 'Content-Type'
app.debug = False
app.url_map.strict_slashes = False

# Function to reinitialize boto clients
def reinitialize_client(region_name):
    logger.info(f"Re-initializing bedrock-runtime client in '{region_name}' region.")
    config = Config(
        connect_timeout=10, 
        read_timeout=10, 
        retries={
            'max_attempts': 2,
            'mode': 'standard'
        }
    )
    client = boto3.client(service_name='bedrock-runtime', region_name=region_name, config=config)
    logger.info("Client reinitialized successfully.")
    return client

def call_llm(client, prompt, model='meta.llama3-8b-instruct-v1:0', max_tokens=1024, sys_prompt="You are a helpful AI assistant which returns in JSON.", max_retries=3, retry_delay=8):
    logger.info("LLM request received")
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": max_tokens,
        "temperature": 0.05,
    }

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"LLM Call {attempt}")
            response = client.invoke_model(
                modelId=model,
                body=json.dumps(native_request),
                contentType="application/json",
                accept="application/json",
                trace='ENABLED'
            )
            logger.info(f"Raw Response: {response}")

            if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 200:
                raise ValueError(f"Unexpected status code received: {response.get('ResponseMetadata', {}).get('HTTPStatusCode')}")

            response_body = response['body'].read().decode('utf-8')
            logger.info(f"Response Data: {response_body}")

            response_json = json.loads(response_body)
            result = response_json.get("generation")
            if not result:
                raise ValueError("No 'generation' field found in the response.")
            return result

        except (ClientError, EndpointConnectionError, ReadTimeoutError) as e:
            logger.error(f"Error during LLM call. Reason: {e}")
            if isinstance(e, ClientError) and e.response['Error']['Code'] == 'ExpiredToken':
                logger.error("Token expired during LLM call. Reinitializing client...")
                client = reinitialize_client(client.meta.region_name)
            else:
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1} of {max_retries})")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Max retries reached. Failed to invoke '{model}'.")
                    return None

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(retry_delay)
            else:
                logger.error(f"Max retries reached. Failed to invoke '{model}'.")
                return None

def create_embeddings(embed_client, text, model='amazon.titan-embed-text-v2:0', max_retries=3, retry_delay=8):
    for attempt in range(max_retries + 1):
        try:
            response = embed_client.invoke_model(
                modelId=model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({'inputText': text})
            )

            response_body = response['body'].read().decode('utf-8')

            embeddings = json.loads(response_body).get("embedding")
            if embeddings is None:
                logger.info(f"Raw Response: {response}")
                logger.info(f"Response Data: {response_body}")
                raise ValueError("No 'embedding' field found in the response.")
            
            return embeddings

        except (ClientError, EndpointConnectionError, ReadTimeoutError) as e:
            logger.error(f"Error during embedding call. Reason: {e}")
            if isinstance(e, ClientError) and e.response['Error']['Code'] == 'ExpiredToken':
                logger.error("Token expired during embedding call. Reinitializing client...")
                embed_client = reinitialize_client(embed_client.meta.region_name)
            else:
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1} of {max_retries})")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Max retries reached. Failed to invoke '{model}'.")
                    return None

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(retry_delay)
            else:
                logger.error(f"Max retries reached. Failed to invoke '{model}'.")
                return None

# def error_response(status_code, message, **kwargs):
#     body = {'error': message}
#     if kwargs:
#         body.update(kwargs)
#     return jsonify(body), status_code
def error_response(status_code, message, details=None, raw_response=None):
    response = {
        'statusCode': status_code,
        'body': {
            'error': message,
            'details': details
        }
    }
    if raw_response:
        response['body']['raw_response'] = raw_response
    return jsonify(response), status_code

# data = {
#     "query": "Hello jinie",
#     "conversation_id" : "abc"
# }

@app.route('/process_query', methods=['POST', 'OPTIONS'])
def process_query():
        logger.info("Flask handler started.")
    # try:
        # Check for the API key in the request headers
        request_api_key = request.headers.get('x-api-key')
        if request_api_key != API_KEY:
            return error_response(401, "Unauthorized: Invalid or missing API key.")
        
        # Parse the incoming JSON data
        data = {
            "query": "Hello jinie",
            "conversation_id" : "abc"
        }
        if not data:
            raise ValueError("Request body is missing or not in JSON format.")
        logger.info(f"Received request data: {data}")
        
        # Retrieve query and conversation_id from the request
        query = data.get('query')
        if not query:
            return error_response(400, "Query parameter is missing or empty.")
        logger.info(f"Received query: {query}")

        conv_id = data.get('conversation_id')
        if not conv_id:
            return error_response(400, "Conversation ID is missing or empty.")
        logger.info(f"Received conversation_id: {conv_id}")
        
        # Retrieve past queries
        logger.info("Retrieving past queries from DocumentDB.")
        past_queries, msg_id = get_past_queries(conv_id, k=4)

        # Set up Bedrock client with timeout configurations
        config = Config(connect_timeout=60, read_timeout=60, retries={'max_attempts': 2, 'mode': 'standard'})
        client = boto3.client(service_name='bedrock-runtime', region_name="ap-south-1", config=config)
        
        # Rewrite the current query using LLM if past queries exist
        if past_queries:
            logger.info(f"Past queries exist: {len(past_queries)}")
            rewrite_prompt = get_rewriting_prompt(query, past_queries)
            logger.info("Rewrite Prompt created successfully.")

            rewrite_response = call_llm(client, rewrite_prompt)
            if rewrite_response:
                try:
                    rewrite_response_data = json.loads(rewrite_response)
                    rewritten_query = rewrite_response_data.get("query", query) if rewrite_response_data.get("need_to_rewrite", False) else query
                    logger.info(f"Rewritten Query: {rewritten_query}")
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON decode error during rewriting: {json_err}")
                    rewritten_query = query
            else:
                logger.error("Failed to get rewritten query from LLM.")
                rewritten_query = query
        else:
            logger.info("No past queries found. Using the original query.")
            rewritten_query = query
        
        # Store the query and rewritten query
        query_id = store_query(conv_id, msg_id, query, rewritten_query)
        logger.info(f"Query stored successfully with ID: {query_id}")

        # Generate embeddings for the rewritten query
        embed_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2", config=config)
        logger.info("Creating embeddings for the query.")
        query_embedding = create_embeddings(embed_client, rewritten_query)
        if not query_embedding:
            logger.error("Failed to create embeddings for the query.")
            return error_response(500, "Failed to generate embeddings.")

        # Search for similar documents in the database
        logger.info("Performing similarity search on DocumentDB.")
        similar_docs = similarity_search(embedding=query_embedding, embedding_key='embedding', text_key='text', k=5)
        logger.info(f"Found similar documents: {len(similar_docs)}")

        # Generate the final prompt to call the LLM
        prompt = get_prompt(rewritten_query, similar_docs, conv_id)
        logger.info("Prompt generated successfully.")

        # Call LLM to get the final response
        logger.info("Calling LLM to generate the response.")
        response = call_llm(client, prompt)
        if not response:
            logger.error("Failed to generate a response from LLM.")
            return error_response(500, "Failed to get a response from LLM.")

        # Parse the response and validate the structure
        try:
            parsed_response = json.loads(response)
            logger.info("LLM response parsed successfully.")
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e}")
            return error_response(500, "Failed to parse LLM response.", raw_response=response)

        # Store the response
        response_id = store_response(conv_id, msg_id, parsed_response)
        logger.info(f"Response stored successfully with ID: {response_id}")

        logger.info("Process completed successfully.")
        return parsed_response.get('References')
        return jsonify({'statusCode': 200, 'body': {'response': parsed_response}})

    # except Exception as e:
    #     logger.error(f"Unexpected error in Flask handler: {e}")
    #     return error_response(500, "An unexpected error occurred.", details=str(e))

if __name__ == '__main__':
    print(process_query())
    # app.run(host='0.0.0.0', port=5004)