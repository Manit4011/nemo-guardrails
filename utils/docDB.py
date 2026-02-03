import os
import json
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from secrets_manager import DocDB_API_URL

load_dotenv()

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Function to insert a single document into the database
def insert_one_entry(document):
    try:
        response = requests.post(
            f"{DocDB_API_URL}/insert_one", 
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME"),
                "document": document
            },
            headers={
                "Content-Type": "application/json"
            }
        )
        
        # Check if the API responded successfully
        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['inserted_id']
        else:
            logger.error(f"API error during insert: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during insert: {e}")
        return None

# Function to insert multiple documents into the database
def insert_many_entries(documents):
    try:
        response = requests.post(
            f"{DocDB_API_URL}/insert_many",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME"),
                "documents": documents
            },
            headers={
                "Content-Type": "application/json"
            }
        )

        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['inserted_ids']
        else:
            logger.error(f"API error during insert: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during insert: {e}")
        return None

# Function to find one document from the database
def find_one_entry(filter_query):
    try:
        response = requests.get(
            f"{DocDB_API_URL}/find_one",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME"),
                "filter_query": filter_query
            },
            headers={
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['document']
        else:
            logger.error(f"API error during find: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during find: {e}")
        return None

# Function to find all documents from the database
def find_all_entries(filter_query={}, projection=None):
    try:
        response = requests.get(
            f"{DocDB_API_URL}/find_all",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME"),
                "filter_query": filter_query,
                "projection": projection
            },
            headers={
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['documents']
        else:
            logger.error(f"API error during find: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during find: {e}")
        return None

# Function to update an entry in the database
def update_entry(filter_query, update_query):
    try:
        response = requests.post(
            f"{DocDB_API_URL}/update",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME"),
                "filter_query": filter_query,
                "update_query": update_query
            },
            headers={
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['update_count']
        else:
            logger.error(f"API error during update: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during update: {e}")
        return None

# Function to delete one entry from the database
def delete_one_entry(filter_query):
    try:
        response = requests.post(
            f"{DocDB_API_URL}/delete_one",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME"),
                "filter_query": filter_query
            },
            headers={
                "Content-Type": "application/json"
            }
        )

        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['delete_count']
        else:
            logger.error(f"API error during delete: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during delete: {e}")
        return None

# Function to delete multiple entries from the database
def delete_many_entries(filter_query={}):
    try:
        response = requests.post(
            f"{DocDB_API_URL}/delete_many",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME"),
                "filter_query": filter_query
            },
            headers={
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['delete_count']
        else:
            logger.error(f"API error during delete: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during delete: {e}")
        return None

# Function to perform similarity search
def similarity_search(embedding, embedding_key, text_key, similarity_type="cosine", k=5, ef_search=50, filter_val=None):
    try:
        response = requests.get(
            f"{DocDB_API_URL}/similarity",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME"),
                "embedding": embedding,
                "embedding_key": embedding_key,
                "text_key": text_key,
                "similarity": similarity_type,
                "k": k,
                "ef_search": ef_search,
                "filter": filter_val or {}
            },
            headers={
                "Content-Type": "application/json"
            }
        )

        if response.status_code == 200:
            logger.info("Similarity search successful")
            result = json.loads(response.text)  # Parse the response as JSON
            if 'docs' in result:
                return result.get('docs')
            else:
                logger.error(f"API response missing 'docs': {response.text}")
                return None
        else:
            logger.error(f"API error during similarity search: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during similarity search: {e}")
        return None


def delete_documents_from_db(object_key):
    return delete_many_entries({'metadata.file_name': object_key})

# Function to retrieve the last k messages for the given conversation ID from the past hour
def get_past_queries(conv_id: str, k: int):
    try:
        response = requests.get(
            f"{DocDB_API_URL}/get_past_queries",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME_2"),
                "conv_id": conv_id,
                "k": k
            },
            headers={
                "Content-Type": "application/json"
            }
        )
        print(response)
        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['past_messages'], int(result['msg_id'])
        else:
            logger.error(f"API error during fetching history: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during fetching history: {e}")
        return None

# Function to store user query to retrieve later
def store_query(conv_id, msg_id, query, rewritten_query):
    try:
        response = requests.post(
            f"{DocDB_API_URL}/store_query",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME_2"),
                "conv_id": conv_id,
                "msg_id": msg_id,
                "query": query, 
                "rewritten_query": rewritten_query
            },
            headers={
                "Content-Type": "application/json"
            }
        )

        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['insert_id']
        else:
            logger.error(f"API error during saving query to history: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during saving query to history: {e}")
        return None

# Function to store response generated to user query to retrieve later
def store_response(conv_id, msg_id, context):
    try:
        response = requests.post(
            f"{DocDB_API_URL}/store_response",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("COLLECTION_NAME_2"),
                "conv_id": conv_id,
                "msg_id": msg_id,
                "context": context
            },
            headers={
                "Content-Type": "application/json"
            }
        )

        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['insert_id']
        else:
            logger.error(f"API error during saving response to history: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during saving response to history: {e}")
        return None
    

def store_conversation(conv_id, msg_id, query, rewritten_query):
    try:
        response = requests.post(
            f"{DocDB_API_URL}/store_query",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("SUMMARY_COLLECTION_NAME"),
                "conv_id": conv_id,
                "msg_id": msg_id,
                "query": query, 
                "rewritten_query": rewritten_query
            },
            headers={
                "Content-Type": "application/json"
            }
        )

        print(response.json())
        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['insert_id']
        else:
            logger.error(f"API error during saving query to history: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during saving query to history: {e}")
        return None
    
def get_past_conversation(conv_id: str, k: int, time_limit: int = None):
    try:
        response = requests.get(
            f"{DocDB_API_URL}/get_past_queries",
            json={
                "database_name": os.getenv("DB_NAME"),
                "collection_name": os.getenv("SUMMARY_COLLECTION_NAME"),
                "conv_id": conv_id,
                "k": k,
                "time_limit": time_limit
            },
            headers={
                "Content-Type": "application/json"
            }
        )
        # print(response)
        if response.status_code == 200:
            logger.info(response.text)
            result = json.loads(response.text) # Parse the response as JSON and return it
            return result['past_messages'], int(result['msg_id'])
        else:
            logger.error(f"API error during fetching history: {response.text}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during fetching history: {e}")
        return None

# Function to retrieve conversation history for the given conversation ID
def get_conversation_history(conv_id=None):
    try:
        filter_query = {}
        if conv_id:
            filter_query = {
                "conv_id": conv_id
            }
        page = 1
        page_size = 500
        total = 1
        data = []
        while(page<=total):
            response = requests.get(
                f"{DocDB_API_URL}/find_all",
                json={
                    "database_name": os.getenv("DB_NAME"),
                    "collection_name": os.getenv("COLLECTION_NAME_2"),
                    "filter_query": filter_query,
                    "projection": None,
                    "page": page,
                    'page_size': page_size
                },
                headers={
                    "Content-Type": "application/json"
                }
            )
            if response.status_code == 200:
                logger.info(response.text)
                result = json.loads(response.text) # Parse the response as JSON and return it
                if 'documents' in result and 'size' in result:
                    data.extend(result['documents'])
                    total = (result['size'] // page_size) + (1 if result['size'] % page_size > 0 else 0)
                else:
                    logger.error(f"Missing 'documents' or 'size' in response: {response.text}")
                    return None
            else:
                logger.error(f"API error during find: {response.text}")
                return None
            page+=1
        if not data:
            logger.info("No records found.")
            return None
        
        df = pd.DataFrame(data)
        df.drop(columns=['_id'], inplace=True, errors='ignore')
        return df
    except Exception as e:
        logger.error(f"An error occurred during find: {e}")
        return None