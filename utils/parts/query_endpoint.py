import json
import boto3
import logging
import time
import os
from botocore.config import Config
from pymongo import MongoClient
from docdbVS import get_past_queries, store_query, store_response, similarity_search
from bedrock_call import get_rewriting_prompt, create_embeddings, get_prompt, call_llm

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Function to reinitialize boto clients
def reinitialize_client(region_name):
    logger.info(f"Re-initializing bedrock-runtime client in '{region_name}' region.")
    config = Config(
        connect_timeout=10, 
        read_timeout=10, 
        retries={
            'max_attempts': 2,
            'mode': 'standard'  # or 'adaptive' for AWS-managed retries
        }
    )
    client = boto3.client(service_name='bedrock-runtime', region_name=region_name, config=config)
    logger.info("Client reinitialized successfully.")
    return client


def error_response(status_code, message, **kwargs):
    body = {'error': message}
    if kwargs:
        body.update(kwargs)
    return {
        'statusCode': status_code,
        'body': body
    }


def lambda_handler(event, context):
    logger.info("Lambda handler started.")
    try:
        logger.info(f"Received event: {json.dumps(event)}")

        query = event.get('query') or json.loads(event.get('body', '{}')).get('query')
        if not query:
            logger.error("Query parameter is missing or empty.")
            return error_response(400, "Query parameter is missing or empty.")
        logger.info(f"Received query: {query}")

        conv_id = event.get('conversation_id') or json.loads(event.get('body', '{}')).get('conversation_id')
        if not conv_id:
            logger.error("Conversation ID is missing or empty.")
            return error_response(400, "Conversation ID is missing or empty.")
        logger.info(f"Received conversation_id: {conv_id}")
        
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        mongo_client = MongoClient(os.environ.get("CONNECTION_STRING"))
        db = mongo_client["PolicyDB"]
        data_collection = db["Test"]
        store_collection = db["Conversation"]
        logger.info("Connected to MongoDB successfully.")

        # Retrieve past queries
        logger.info("Retrieving past queries from DocumentDB.")
        past_queries, msg_id = get_past_queries(store_collection, conv_id, k=4)

        # Set up Bedrock client with timeout configurations
        config = Config(
            connect_timeout=10, 
            read_timeout=10, 
            retries={
                'max_attempts': 2,
                'mode': 'standard'  # or 'adaptive' for AWS-managed retries
            }
        )
        client = boto3.client(service_name='bedrock-runtime', region_name="ap-south-1", config=config)
        
        # Rewrite the current query using LLM if past queries exist
        if past_queries:
            logger.info(f"Past queries exist: {len(past_queries)}")

            logger.info("Creating prompt for rewriting query.")
            rewrite_prompt = get_rewriting_prompt(query, past_queries)
            logger.info("Rewrite Prompt created successfully.")

            logger.info("Calling the LLM to rewrite the query.")
            rewrite_response = call_llm(client, rewrite_prompt)
            if rewrite_response:
                rewrite_response_data = json.loads(rewrite_response)
                logger.info(f"Rewrite Response: {rewrite_response_data}")
                rewritten_query = rewrite_response_data.get("query", query) if rewrite_response_data.get("need_to_rewrite", False) else query
                logger.info(f"Rewritten Query: {rewritten_query}")
            
            else:
                logger.error("Failed to get rewritten query from LLM.")
                rewritten_query = query
            
        else:
            logger.info("No past queries found. Using the original query.")
            rewritten_query = query

        # Store the query and rewritten query
        query_id = store_query(store_collection, conv_id, msg_id, query, rewritten_query)
        logger.info(f"Query stored successfully with ID: {query_id}")

        
        # Generate embeddings for the rewritten query
        embed_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2", config=config)
        logger.info("Creating embeddings for the query.")
        query_embedding = create_embeddings(embed_client, rewritten_query)
        if not query_embedding:
            logger.error("Failed to create embeddings for the query.")
            return error_response(500, "Failed to create embeddings for the query.")
        logger.info(f"Query Embedding created successfully with length: {len(query_embedding)}")

        # Perform similarity search using the generated embeddings
        logger.info("Performing similarity search in DocumentDB.")
        top_k_results = similarity_search(
            collection=data_collection,
            embedding=query_embedding,
            embedding_key='embedding',
            text_key='text',
            k=5
        )
        if not top_k_results:
            logger.error("No results returned from similarity search.")
            return error_response(404, "No results returned from similarity search.")
        logger.info(f"Top K results aggregated successfully (k: {len(top_k_results)})")
        
        # Create the prompt for the LLM
        logger.info("Creating prompt for the LLM.")
        prompt = get_prompt(rewritten_query, top_k_results, conv_id)
        logger.info("Prompt created successfully.")
        
        # Call the LLM to generate a response
        logger.info("Calling the LLM to generate a response.")
        response = call_llm(client, prompt)
        logger.info(f"LLM response received: {response}")
        
        if not response:
            logger.error("Failed to get response from LLM.")
            return error_response(500, "Failed to get response from LLM.")
        
        # Ensure the response is a valid JSON string
        try:
            parsed_response = json.loads(response)
            logger.info("LLM response parsed successfully.")
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e}")
            return error_response(500, "Failed to parse LLM response.", raw_response=response)

        # Store the response
        response_id = store_response(store_collection, conv_id, msg_id, parsed_response)
        logger.info(f"Response stored successfully with ID: {response_id}")

        logger.info("Lambda handler completed successfully.")
        return {
            'statusCode': 200,
            'body': {'response': parsed_response}
        }

    except (KeyError, ValueError, EnvironmentError) as e:
        logger.error(f"Error: {str(e)}")
        return error_response(400, str(e))

    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        return error_response(500, "An unexpected error occurred.", details=str(e))