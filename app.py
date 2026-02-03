from flask import Flask, request, jsonify, Response
import json
import boto3
import logging
import time
import os
import pandas as pd
from urllib.parse import unquote_plus
from dotenv import load_dotenv
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from utils.bucket_call import generate_presigned_pdf_url
from utils.docDB import get_past_queries, store_query, store_response, similarity_search, store_conversation, get_past_conversation
from utils.bedrock_call import call_llm_sonet, create_embeddings, get_prompt, get_rewriting_prompt, clean_raw_response, get_summary_prompt
from utils.feedback.analyze import update_conversation_history, analyze_history, get_ragas_analysis, analyse_user_sentiment
from secrets_manager import API_KEY, PROD_CHANNEL_ID, PROD_CHANNEL_ID2, ITSM_KEY, ITSM_CHANNEL_ID
import requests
import random

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

def call_backend_api(url, headers, data):
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error calling backend API: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        logger.error(f"RequestException: {e}")
        return None
    
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

def generate_urls(policies):
    urls = []     
    for policy in policies:
        if(policy!="FAQ"):
            temp = policy+".pdf"
            url = generate_presigned_pdf_url(os.getenv("S3_BUCKET"), temp)
            urls.append({policy: url})
    return urls

def generate_urls1(policies):
    urls = []
    for policy in policies.get('PDF', []):
        temp = policy+".pdf"
        url = generate_presigned_pdf_url(os.getenv("S3_BUCKET"), temp)
        urls.append({policy: url})
    for subject in policies:
        if(subject!="FAQ" and subject!="PDF"):
            url = policies[subject]
            urls.append({subject: unquote_plus(url)})
    return urls

# data = {
#     "query": "Hello jinie",
#     "conversation_id" : "abc"
# }

@app.route('/', methods=['GET'])
def api_check():
    return jsonify({'status': 200, 'message': "PASS"}), 200

@app.route('/update_history', methods=['GET'])
def update_history_route():
    # Check for the API key in the request headers
    request_api_key = request.headers.get('x-api-key')
    if request_api_key != API_KEY:
        return error_response(401, "Unauthorized: Invalid or missing API key.")
    response = update_conversation_history()
    if not response:        
        return jsonify({'status': 'success', 'message': 'Conversation history updated successfully'}), 200
    else:
        return jsonify({'status': 'error', 'message': response}), 500

# 2. Route for analyzing history
@app.route('/analyze_history', methods=['GET'])
def analyze_history_route():
    try:
        df = analyze_history(file_name='conv_history.xlsx')
        return jsonify({'status': 'success', 'message': 'Conversation history analysis complete'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# 3. Route for performing RAGAS analysis
@app.route('/ragas_analysis', methods=['GET'])
def ragas_analysis_route():
    try:
        df = get_ragas_analysis(file_name='conv_history.xlsx') 
        return jsonify({'status': 'success', 'message': 'RAGAS analysis completed successfully'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 4. Route for analyzing user sentiment
@app.route('/analyze_sentiment', methods=['GET'])
def analyze_sentiment_route():
    try:
        df = pd.read_excel('conv_history.xlsx')  # Load data
        sentiment_data = analyse_user_sentiment(df) 
        return jsonify({'status': 'success', 'message': 'Sentiment analysis complete', 'sentiment': sentiment_data.tolist()}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


def process_query_helper(conv_id, query):
    # Retrieve past queries
    logger.info("Retrieving past queries from DocumentDB.")
    past_queries, msg_id = get_past_queries(conv_id, k=4)

    # Set up Bedrock client with timeout configurations
    config = Config(connect_timeout=60, read_timeout=60, retries={'max_attempts': 2, 'mode': 'standard'})
    client = boto3.client(service_name='bedrock-runtime', region_name=os.getenv("REGION_NAME_1"), config=config)

    # Rewrite the current query using LLM if past queries exist
    if past_queries:
        logger.info(f"Past queries exist: {len(past_queries)}")
        rewrite_prompt = get_rewriting_prompt(query, past_queries)
        logger.info("Rewrite Prompt created successfully.")

        rewrite_response = call_llm_sonet(client, rewrite_prompt, max_tokens = 512)
        if rewrite_response:
            try:
                rewrite_response_data = json.loads(rewrite_response)
                rewritten_query = rewrite_response_data.get("query", query) if rewrite_response_data.get("need_to_rewrite", False) else query
                logger.info(f"Rewritten Query: {rewritten_query}")
            except json.JSONDecodeError as json_err:
                if isinstance(rewrite_response, str) and rewrite_response.startswith("Content Restricted:"):
                    restricted_message = rewrite_response[len("Content Restricted:"):].strip()
                    parsed_response = {
                        "Answer": restricted_message,
                        "urls":[]
                    }                
                    query_id = store_query(conv_id, msg_id, query, query)
                    response_id = store_response(conv_id, msg_id, parsed_response)
                    return jsonify({'statusCode': 200, 'body': {'response': parsed_response}})
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
    embed_client = boto3.client(service_name='bedrock-runtime', region_name=os.getenv("REGION_NAME_1"), config=config)
    logger.info("Creating embeddings for the query.")
    query_embedding = create_embeddings(embed_client, rewritten_query)
    if not query_embedding:
        logger.error("Failed to create embeddings for the query.")
        return error_response(500, "Failed to generate embeddings.")
    
    # Search for similar documents in the database
    logger.info("Performing similarity search on DocumentDB.")
    similar_docs = similarity_search(embedding=query_embedding, embedding_key='embedding', text_key='text', k=5)
    logger.info(f"Found similar documents: {len(similar_docs)}")
    logger.info(f"Similar documents: {similar_docs}")

    # Generate the final prompt to call the LLM
    prompt = get_prompt(rewritten_query, similar_docs, conv_id)
    logger.info("Prompt generated successfully.")
    logger.info(f"Prompt: {prompt}")

    
    # Call LLM to get the final response
    logger.info("Calling LLM to generate the response.")
    response = call_llm_sonet(client, prompt)
    if not response:
        logger.error("Failed to generate a response from LLM.")
        return error_response(500, "Failed to get a response from LLM.")
    

            # Parse the response and validate the structure
    try:
        parsed_response = json.loads(response)
        logger.info("LLM response parsed successfully.")
        logger.info(f"Parsed response: {parsed_response}")
    except json.JSONDecodeError as e:
        if isinstance(response, str) and response.startswith("Content Restricted:"):
            restricted_message = response[len("Content Restricted:"):].strip()
            parsed_response = {
                "Answer": restricted_message,
                "urls": []
            }                
            query_id = store_query(conv_id, msg_id, query, rewritten_query)
            response_id = store_response(conv_id, msg_id, parsed_response)
            return jsonify({'statusCode': 200, 'body': {'response': parsed_response}})
        logger.error(f"JSONDecodeError: {e}")
        try:
            logger.info(f"Raw response: {response}")
            logger.info("Cleaning the raw response.")
            result = clean_raw_response(client, response)
            parsed_response = json.loads(result)
        except json.JSONDecodeError as e:
            return error_response(500, "Failed to parse LLM response.", raw_response=response)
        
    # Generate links for policies
    parsed_response["urls"] = generate_urls(parsed_response.get("Policies", {}))
    logger.info(f"Links generated successfully for the policies")

    # Store the response
    response_id = store_response(conv_id, msg_id, parsed_response)
    logger.info(f"Response stored successfully with ID: {response_id}")

    logger.info("Process completed successfully.")
    return jsonify({'statusCode': 200, 'body': {'response': parsed_response}})



def summary_helper(conv_id, query):
    # Placeholder for summary processing logic
    if any(word in query.lower() for word in ["summary", "summarize", "summarise"]):
        logger.info("Generating summary...")
        config = Config(connect_timeout=60, read_timeout=60, retries={'max_attempts': 2, 'mode': 'standard'})
        client = boto3.client(service_name='bedrock-runtime', region_name=os.getenv("REGION_NAME_1"), config=config)

        query_arr = query.split(" ")
        if len(query_arr) > 1:
            k = int(query_arr[1])
        else:
            k = 18

        past_conv, msg_id =  get_past_conversation(conv_id, k, time_limit=24)
        conv_arr = [conv['query'] for conv in sorted(past_conv, key=lambda x: int(x['msg_id']))]
        summary_prompt = get_summary_prompt(conv_arr, k)

        logger.info(f"Past conversation {len(past_conv)}")
        logger.info(f"Summary Prompt: {summary_prompt}")

        # return error_response(200, "Summary is not available yet.", details="Summary generation is not implemented yet.")

       
        logger.info("Calling LLM to generate the response.")
        response = call_llm_sonet(client, summary_prompt)
        if not response:
            logger.error("Failed to generate a response from LLM.")
            return error_response(500, "Failed to get a response from LLM.")
    

            # Parse the response and validate the structure
        try:
            parsed_response = json.loads(response)
            logger.info("LLM response parsed successfully.")
            logger.info(f"Parsed response: {parsed_response}")
        except json.JSONDecodeError as e:
            if isinstance(response, str) and response.startswith("Content Restricted:"):
                restricted_message = response[len("Content Restricted:"):].strip()
                parsed_response = {
                    "Answer": restricted_message,
                    "urls": []
                }                
                # query_id = store_query(conv_id, msg_id, query, rewritten_query)
                # response_id = store_response(conv_id, msg_id, parsed_response)
                return jsonify({'statusCode': 200, 'body': {'response': parsed_response}})
            logger.error(f"JSONDecodeError: {e}")
            try:
                logger.info(f"Raw response: {response}")
                logger.info("Cleaning the raw response.")
                result = clean_raw_response(client, response)
                parsed_response = json.loads(result)
            except json.JSONDecodeError as e:
                return error_response(500, "Failed to parse LLM response.", raw_response=response)

        logger.info("Process completed successfully.")
        return jsonify({'statusCode': 200, 'body': {'response': parsed_response}})

        


    else:
        past_queries, msg_id = get_past_conversation(conv_id, k=2, time_limit=24)
        logger.info(f"Past msg id: {msg_id}")
        store_conv_res = store_conversation(conv_id, msg_id, query, query)
        if not store_conv_res:
            logger.error("Failed to store conversation.")
            return error_response(500, "Failed to store conversation.")
        logger.info(f"Conversation stored successfully with ID: {store_conv_res}")
        return jsonify({'statusCode': 200, 'body': {'response': "Conversation stored successfully."}})

    

@app.route('/process_query', methods=['POST', 'OPTIONS'])
def process_query():
    logger.info("Flask handler started.")
    try:
        # Check for the API key in the request headers
        request_api_key = request.headers.get('x-api-key')
        if request_api_key != API_KEY:
            return error_response(401, "Unauthorized: Invalid or missing API key.")
        
        # Parse the incoming JSON data
        data = request.get_json()
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
        
        
        # if conv_id == ITSM_CHANNEL_ID:
        #     logger.info("Processing ITSM query...")
        #     payload = {
        #         "message": {
        #         "subject": "",
        #         "body": {
        #             "contentType": "text",
        #             "content": query
        #             }
        #         }
        #     }

        #     headers = {
        #         'Content-Type': 'application/json',
        #         'x-api-key': ITSM_KEY
        #     }
        #     try:
        #         api_response  = call_backend_api("http://52.66.10.81:5000/process_query", headers, payload)
        #         print(api_response)
        #         return api_response
        #     except Exception as e:
        #         logger.error(f"Error calling ITSM API: {e}")
        #         return error_response(500, "Failed to call ITSM API.", details=str(e))
        
        if conv_id == PROD_CHANNEL_ID or conv_id == PROD_CHANNEL_ID2:
            logger.info("Processing summary helper...")
            summary_response = summary_helper(conv_id, query)
            return summary_response

        else:
            logger.info("Processing query helper...")
            query_helper_response = process_query_helper(conv_id, query)
            return query_helper_response


    except Exception as e:
        logger.error(f"Unexpected error in Flask handler: {e}")
        return error_response(500, "An unexpected error occurred.", details=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)