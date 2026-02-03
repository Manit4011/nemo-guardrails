import os
import boto3
import json
import time
import logging
from botocore.config import Config
from urllib.parse import quote
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from dotenv import load_dotenv
from secrets_manager import prompt_guardrail_arn

load_dotenv()
# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialising boto clients
# client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")
# embed_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

# Function to reinitialize boto clients
def reinitialize_client(region_name):
    logger.info(f"Re-initializing bedrock-runtime client in '{region_name}' region.")
    config = Config(connect_timeout=60, read_timeout=60, retries={'max_attempts': 2, 'mode': 'standard'})
    client = boto3.client(service_name='bedrock-runtime', region_name=region_name, config=config)
    logger.info("Client reinitialized successfully.")
    return client

def clean_raw_response(client, raw_response):
    try:
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith('"""') and cleaned_response.endswith('"""'):
            cleaned_response = cleaned_response.strip('"""')
            response = json.loads(cleaned_response)
        return response
    except Exception as e:
        prompt = f"You need to correct the JSON format of this raw_response to be parsed successfully. raw_response:{raw_response}"
        response = call_llm_sonet(client, prompt)
        return response

def call_llm_sonet(client, prompt: str, guardrail: str = prompt_guardrail_arn,
                    model: str ="anthropic.claude-3-5-sonnet-20240620-v1:0",
                    sys_prompt="You are a helpful AI assistant which returns in JSON.", 
                    max_tokens=1024, temperature=0, max_retries=3, retry_delay=8
                ) -> str:
    logger.info("LLM request received")
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": formatted_prompt}],
            }
        ],
    }
    request = json.dumps(native_request)
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"LLM Call {attempt}")
            invoke_params = {"modelId": model, "body": request}
            if guardrail:
                logger.info("Guardrail on Model Call")
                invoke_params["guardrailIdentifier"] = guardrail
                invoke_params["guardrailVersion"] = "3"
            response = client.invoke_model(**invoke_params)
            model_response = json.loads(response["body"].read())
            if not model_response:
                logger.info(f"Raw Response: {response}")
                raise ValueError("Issue with json.loads.")
            response_text = model_response["content"][0]["text"]
            if not response_text:
                logger.info(f"Response Data: {model_response}")
                raise ValueError("No 'content text' field found in the response.")
            return response_text
        except (ClientError, EndpointConnectionError, ReadTimeoutError) as e:
            logger.error(f"Error during LLM call. Reason: {e}")
            if isinstance(e, ClientError) and e.response['Error']['Code'] == 'ExpiredToken':
                logger.error("Token expired during LLM call. Reinitializing client...")
                client = reinitialize_client("us-west-2")
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

def call_llm(client, prompt, model='meta.llama3-70b-instruct-v1:0', max_tokens=1024, sys_prompt = "You are a helpful AI assistant which returns in JSON."):
    logger.info("LLM request received")
    
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": max_tokens,
        "temperature": 0.1,
    }
    try:
        response = client.invoke_model(
            modelId=model, 
            body=json.dumps(native_request)
        )
        logger.info(f"Raw Response: {response}")
        
        # Check for HTTP status code in response
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 200:
            raise ValueError(f"Unexpected status code received: {response.get('ResponseMetadata', {}).get('HTTPStatusCode')}")

        # Extract and decode the response body
        response_body = response.get("body")
        if not response_body:
            raise ValueError("Empty response body received from Bedrock runtime.")

        response_data = response_body.read().decode('utf-8')
        logger.info(f"Response Data: {response_data}")

        # Parse the response data
        response_json = json.loads(response_data)
        result = response_json.get("generation")
        if not result:
            raise ValueError("No 'generation' field found in the response.")
        return result

    except (ClientError, Exception) as e:
        logger.error(f"ERROR: Can't invoke '{model}'. Reason: {e}")
        return None


def call_llama(client, prompt, model='meta.llama3-1-70b-instruct-v1:0', max_tokens=1024):
    logger.info("LLM request received")
    # Start a conversation with the user message.
    conversation = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    try:
        client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")
        response = client.converse(
            modelId=model,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2, "topP": 0.9},
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text

    except (ClientError, Exception) as e:
        logger.error(f"ERROR: Can't invoke '{model}'. Reason: {e}")
        exit(1)
        

def get_rewriting_prompt(current_query, past_history):
    prompt = f"""You are an assistant named Jinie, helping a user by reviewing their queries in the context of their past interactions.
Your task is to evaluate if the current query needs rewriting based on the conversation history to improve relevance and coherence.
Current User Query: {current_query}
Below is the conversation history:"""
    
    for chat in past_history:
        if chat['role'] == 'user':
            prompt += f"User's Query: {chat['query']}\nRewritten Query: {chat['rewritten_query']}\n"
        elif chat['role'] == 'assistant':
            prompt += f"Assistant's Response: {chat['response']}\nReference: {chat['reference']}\n"    
    prompt += """
Your goal is to assess whether the current query is relevant and coherent in the context of the conversation history. \
If the user query seems irrelevant, overly casual, or nonsensical (e.g., gibberish), do not rewrite the query. \
If the query is appropriate but could be improved for clarity, coherence, or specificity, rewrite it to make it more relevant to the conversation.
Structure your answer as a JSON object with the following keys:
{
    "need_to_rewrite": true/false,
    "query": "<rewritten_query>"
}
Ensure to return only the JSON object without any additional text.
"""
    return prompt


def get_prompt_with_itsm(query, top_k_results, conv_id):
    chunks = list()
    for idx, result in enumerate(top_k_results):
        temp_str = f"Text Block {idx+1}:\n- **Text:** {result.get('text', 'N/A')}\n- "
        metadata = result.get('metadata', {})
        doc_type = result.get('doc_type')
        if doc_type=='pdf':
            temp_str += f"**File Name:** {metadata.get('filename', metadata.get('file_name', 'N/A'))}\n- **Page No:** {metadata.get('page_no', 'N/A')}\n- **Chunk Number:** {metadata.get('chunk_no', 'N/A')}"
        elif doc_type=='xlsx' or doc_type=='csv':
            temp_str += f"**File Name:** {metadata.get('filename', metadata.get('file_name', 'N/A'))}\n- **Sheet Name:** {metadata.get('sheet_name', 'N/A')}\n- **Index:** {metadata.get('index', 'N/A')}"
        elif doc_type=='conv':
            temp_str += f"**File Name:** {metadata.get('file_name', metadata.get('file_name', 'N/A'))}\n- **Sender Name:** {metadata.get('senderName', 'N/A')}\n- **Web Url:** {json.dumps(quote(metadata.get('thread_web_url', 'N/A'), safe=':/?&='))}"
        chunks.append(temp_str)
    chunks = "\n".join(chunks)

    prompt = f"""
You are a bot specialized to answer HR and ITSM related queries. These are the relevant texts delimited in triple backticks retrieved from the database, \
which can either be text from a PDF or a QA pair from an FAQ (in format "Question: X, Answer: Y") or a Conversation Thread from ITSM Query Resolving Teams channel (in format "Subject: A, Message: B"). \
Based on the relevant texts, generate an accurate and clear response to the following user query:
`{query}`

Relevant Texts:
```{chunks}```

If the user query is irrelevant or casual conversation, respond politely to the conversation and mention that you are a bot who can reply to queries regarding company policies and IT related queries only.

If the query is relavant to ITSM then ensure to:
- Carefully review the entire message history to provide the most accurate and contextually relevant response.
- If metadata is identical across messages, avoid repeating identical references.
- If a query contains an image, please ignore it and focus only on the text-based context.
- If you think request from similar past requests are similar to user's query, summarize them in brief (upto 1-2 lines).\
- If no relevant past history is found, acknowledge this and generate a response based on your understanding and say that someone from the team will get back to them. Be polite and concise.

If you cannot find any relevant information from the provided texts to answer the query, ensure to:
- State that no relevant response was found and offer help with anything else, within your scope of domain.

Structure your answer as a JSON object with the following keys:
{{
    "Answer": "<detailed answer in concise format using `*` for points if applicable>",
    "References": [
        {{
            "Statement": "<Relevant sentence from the text block>",
            "source": "<filename - page_no/sheet_name - index as per applicable>"
            "msg_id": "<Unique Message id of the message if applicable>",
            "msg": "<Subject of the message if subject is missing return the relevant phrase from the message if applicable>",
            "link": "<URL to the message thread (web url) if its from ITSM>"
        }}
    ],
    "Policies":{{
        "PDF":[
            "Return the filenames of the policies of PDF file format used for answering the query(trim the .pdf at the end from the filename)."
        ],
        "FAQ": "If the relevant texts are from a CSV or XLSX file, just add 'FAQ' to the list of Policies instead of the filename.",
        "<Subject>":{
            "If the relevant texts are from a conversation thread, return the Web Url of the conversation."
        }
    }},
    "Metadata": {{
        "conversation_id": "{conv_id}",
        "query": "{query}",
        "source": "<top reference source, if applicable>",
        "response_length": "<response length>",
        "ticket_id": <if query is related to ITSM, generate a random ticket id INC-<random 9 digit number> (for e.g. INC-000096832)> 
    }}
  ]
}}

**Important Instructions:**
- Ensure all keys and string values in the JSON are enclosed in double quotes.
- Escape any special characters such as double quotes (`"`) and backslashes (`\\`) within string values.
- Replace newline characters with `\\n` to ensure they are properly formatted in JSON.
- Provide a valid JSON output only without any additional text or notes.

Example Output:
{{
    "Answer": "Your answer here.",
    "References": [
        {{
            "Statement": "This is a relevant sentence from the policy document.",
            "source": "example.pdf - 2"
        }}
    ],
    "Policies":[
        "example"
    ],
    "Metadata": {{
        "conversation_id": "{conv_id}",
        "query": "{query}",
        "source": "example.pdf",
        "response_length": "50"
    }}
}}
"""
    return prompt


def get_summary_prompt(past_conv_arr, k=10):
    
    prompt = f"""
You are an intelligent and helpful assistant for incident management.
You will be given a conversation thread of {k} chats between users discussing a production issue. Your task is to extract and summarize the key information based on the conversation. 

Below is the conversation thread:"""
    i = 1
    for chat in past_conv_arr:
        # if chat['role'] == 'user':
            # prompt += f"User{i}'s Message: {chat['query']}\n"
        prompt += f"User{i}'s Message: {chat}\n"
            
        i+=1

    prompt += f"""  
    Please output your summary in JSON format as given below:
{{ 
    "Answer" : {{
        "Description" : <Brief summary of the issue as discussed in the thread.>
        "Root Cause" : <Clearly explain the underlying cause of the incident, if identified.>
        "Mitigation" : <Describe the actions taken to resolve or mitigate the problem.>
        "Current State" : <State the current status of the incident>
        "Duration" : <Mention how long the issue lasted or is expected to last.>
        "Impact" : <Summarize the effects of the issue on users, systems, or business operations.>
        "Next Steps" : <Recommend any follow-up actions, including improvements, documentation updates, or preventative measures.>
    }}
}}

**Important Instructions:**
- Ensure all keys and string values in the JSON are enclosed in double quotes.
- Escape any special characters such as double quotes (`"`) and backslashes (`\\`) within string values.
- Write the descriptions of each key in a concise manner in a single string (Do not return response in array or bullet points).
- Replace newline characters with `\\n` to ensure they are properly formatted in JSON.
- Provide a valid JSON output only without any additional text or notes.

"""
    return prompt




def get_prompt(query, top_k_results, conv_id):
    chunks = "\n".join([
        f"Text Block {idx+1}:\n- **Text:** {result.get('text', 'N/A')}\n- **File Name:** {metadata.get('filename', metadata.get('file_name', 'N/A'))}\n- **Page No:** {metadata.get('page_no', 'N/A')}\n- **Sheet Name:** {metadata.get('sheet_name', 'N/A')}\n- **Index:** {metadata.get('index', 'N/A')}"
        for idx, result in enumerate(top_k_results)
        if (metadata := result.get('metadata', {}))
    ])

    prompt = f"""
You are an HR policy bot for Gemini Solutions. These are the relevant texts delimited in triple backticks retrieved from the database, \
which can either be text from a PDF or a QA pair from an FAQ (in format "Question: X, Answer: Y"). \
Based on the relevant texts, generate an accurate and clear response to the following user query:
`{query}`

Relevant Texts:
```{chunks}```

If the user query is irrelevant or casual conversation, respond politely to the conversation and mention that you are a bot who can reply to queries regarding company policies.
If you cannot find any relevant information from the provided texts to answer the query, ensure to:
- State that no relevant policy was found and offer help with anything else, within the scope of policies.

Structure your answer as a JSON object with the following keys:
{{
    "Answer": "<detailed answer in concise format using `*` for points if applicable>",
    "References": [
        {{
            "Statement": "<Relevant sentence from the text block>",
            "source": "<filename - page_no/sheet_name - index as per applicable>"
        }}
    ],
    "Policies":[
        "Return the filenames of the policies of PDF file format used for answering the query(trim the .pdf at the end from the filename)",
        "If the relevant texts are from a CSV or XLSX file, just add 'FAQ' to the list of policies instead of the file name"
    ],
    "Metadata": {{
        "conversation_id": "{conv_id}",
        "query": "{query}",
        "source": "<top reference source, if applicable>",
        "response_length": "<response length>"
    }}
}}

**Important Instructions:**
- Ensure all keys and string values in the JSON are enclosed in double quotes.
- Escape any special characters such as double quotes (`"`) and backslashes (`\\`) within string values.
- Replace newline characters with `\\n` to ensure they are properly formatted in JSON.
- Provide a valid JSON output only without any additional text or notes.

Example Output:
{{
    "Answer": "Your answer here.",
    "References": [
        {{
            "Statement": "This is a relevant sentence from the policy document.",
            "source": "example.pdf - 2"
        }}
    ],
    "Policies":[
        "example"
    ],
    "Metadata": {{
        "conversation_id": "{conv_id}",
        "query": "{query}",
        "source": "example.pdf",
        "response_length": "50"
    }}
}}
"""
    return prompt


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