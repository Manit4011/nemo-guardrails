import json
import logging
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
            
            if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 200:
                logger.info(f"Raw Response: {response}")
                raise ValueError(f"Unexpected status code received: {response.get('ResponseMetadata', {}).get('HTTPStatusCode')}")

            response_body = response['body'].read().decode('utf-8')

            response_json = json.loads(response_body)
            result = response_json.get("generation")
            if not result:
                logger.info(f"Response Data: {response_body}")
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

def call_llm_sonet(client, prompt : str, model: str ="anthropic.claude-3-5-sonnet-20240620-v1:0", max_tokens=1024, sys_prompt="You are a helpful AI assistant which returns in JSON.", max_retries=3, retry_delay=8,
    temperature=0,
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
    try:
        response = client.invoke_model(modelId=model, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model}'. Reason: {e}")
        exit(1)

    model_response = json.loads(response["body"].read())
    response_text = model_response["content"][0]["text"]
    return response_text


def get_rewriting_prompt(current_query, past_history):
    prompt = """You are an assistant named Jinie by Gemini, helping a user by reviewing their queries in the context of their past interactions.
Your task is to determine if the current query needs rewriting based on the provided history to enhance relevance and coherence.
Below is the conversation history:"""
    
    for chat in past_history:
        if chat['role'] == 'user':
            prompt += f"User's Query: {chat['query']}\nRewritten Query: {chat['rewritten_query']}\n"
        elif chat['role'] == 'assistant':
            prompt += f"Assistant's Response: {chat['response']}\nReference: {chat['reference']}\n"

    prompt += f"\nCurrent Query: {current_query}\n"
    
    prompt += """
Please assess whether the current query requires rewriting based on the conversation history.
If rewriting is needed, provide the revised version of the current query. If no rewrite is necessary, return the current query as is.
Format your response as a JSON object with the following structure:
{
    "need_to_rewrite": true/false,
    "query": "<rewritten_query>"
}
Ensure to return only the JSON object without any additional text.
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

If the user query is irrelevant or casual conversation, respond politely and mention that you are a bot who can only solve queries regarding company policies.
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