from flask import Flask, request, jsonify, Response, send_file
import pandas as pd
import json
import boto3
import logging
from werkzeug.utils import secure_filename
from botocore.config import Config
from io import BytesIO
from utils.docDB import similarity_search
from utils.bedrock_call import create_embeddings
from app import error_response

app = Flask(__name__)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

def get_feedback_prompt(row, top_k_results):
    # Format the relevant text chunks from the database results
    chunks = "\n".join([
        f"Text Block {idx+1}:\n- **Text:** {result.get('text', 'N/A')}\n- **File Name:** {metadata.get('filename', metadata.get('file_name', 'N/A'))}\n- **Page No:** {metadata.get('page_no', 'N/A')}\n- **Sheet Name:** {metadata.get('sheet_name', 'N/A')}\n- **Index:** {metadata.get('index', 'N/A')}"
        for idx, result in enumerate(top_k_results)
        if (metadata := result.get('metadata', {}))
    ])
    
    # Construct the prompt for the feedback loop
    prompt = f"""
You are Jinie, the HR policy bot for Gemini Solutions. Below are relevant text blocks retrieved from the database. They may come from PDFs or FAQs (formatted as "Question: X, Answer: Y"). \
Your task is to generate an accurate and clear response to the following user query based on the provided texts:
`{row['Question']}`

Relevant Texts:
```{chunks}```

The initial response was insufficient, and feedback has been provided. Please use the feedback to revise the response.

Feedback:
`{row['Remarks']}`

Previous Response:
`{row['Answers Received from chatbot']}`

Generate a concise and accurate response that addresses the user's query. If the query involves steps, criteria, or key factors, present them in bullet points or highlight important items in bold. \
If the query is irrelevant or chitchat, politely inform the user that you're a bot specializing in company policies.

Output the final response as a JSON object using the format below:
{{
    "Answer": "<concise, accurate answer with `*` for bullet points if necessary>"
}}

Return only the JSON object as output, with no additional text."""
    
    return prompt

API_KEY = "Gemini-123"
@app.route('/get_response_after_feedback', methods=['POST'])
def get_response_from_feedback():
    logger.info("Flask handler started.")
    try:
        # Check for the API key in the request headers
        request_api_key = request.headers.get('x-api-key')
        if request_api_key != API_KEY:
            return error_response(401, "Unauthorized: Invalid or missing API key.")
        
        # Check for file in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Determine if the file is CSV or Excel based on the extension
        filename = secure_filename(file.filename)
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(file, encoding='utf-8')  # First attempt with UTF-8
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding='ISO-8859-1')
            # df = pd.read_csv(file)  # Load CSV file into a pandas DataFrame
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file)  # Load Excel file into a pandas DataFrame
        else:
            return jsonify({'error': 'Unsupported file type. Please upload a .csv or .xlsx file.'}), 400

        # Check if 'Question' column is present
        if 'Question' not in df.columns:
            return jsonify({'error': "File must contain a 'Question' column."}), 400
        
        # Set up Bedrock client with timeout configurations
        config = Config(connect_timeout=10, read_timeout=10, retries={'max_attempts': 2, 'mode': 'standard'})
        client = boto3.client(service_name='bedrock-runtime', region_name="ap-south-1")
        embed_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

        # Iterate through the DataFrame and generate responses (same as before)
        responses = []
        for index, row in df.iterrows():
            query = row['Question']
            embedding = create_embeddings(embed_client, query)
            if not embedding:
                return jsonify({'error': "Failed to generate embeddings."}), 500
            similar_docs = similarity_search(embedding=embedding, embedding_key='embedding', text_key='text', k=5)
            logger.info(f"Found similar documents: {len(similar_docs)}")

            # Generate the final prompt to call the LLM
            prompt = get_feedback_prompt(row, similar_docs)
            logger.info("Prompt generated successfully.")

            # Call LLM to get the final response
            logger.info("Calling LLM to generate the response.")
            response = call_llm(client, prompt)
            try:
                parsed_response = json.loads(response)
                if parsed_response:
                    responses.append(parsed_response.get("Answer"))
                else:
                    responses.append(response if response else "No response generated")
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError: {e}")
                responses.append(response if response else "No response generated")

        # Add the responses to a new column in the DataFrame
        df['Answer'] = responses

        # Save the updated DataFrame to a new Excel file in memory
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        return send_file(output, download_name="response_chatbot.xlsx", as_attachment=True)

    except Exception as e:
        logger.error(f"Error generating answers: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_answer', methods=['POST'])
def get_answer():
    logger.info("Flask handler started.")
    try:
        # Check for the API key in the request headers
        request_api_key = request.headers.get('x-api-key')
        if request_api_key != API_KEY:
            return jsonify({'error': "Unauthorized: Invalid or missing API key."}), 401
        
        # Parse the incoming JSON data
        data = request.get_json()
        if not data:
            raise ValueError("Request body is missing or not in JSON format.")
        logger.info(f"Received request data: {data}")
        
        # Retrieve query and conversation_id from the request
        row = data.get('row')
        query = row['Question']
        if not query:
            return error_response(400, "Query parameter is missing or empty.")
        logger.info(f"Received query: {query}")

        # Initialize boto3 client
        client = boto3.client(service_name='bedrock-runtime', region_name="ap-south-1")
        embed_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

        # Set up Bedrock client with timeout configurations
        config = Config(connect_timeout=10, read_timeout=10, retries={'max_attempts': 2, 'mode': 'standard'})
        client = boto3.client(service_name='bedrock-runtime', region_name="ap-south-1")
        embed_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

        embedding = create_embeddings(embed_client, query)
        if not embedding:
            return jsonify({'error': "Failed to generate embeddings."}), 500
        similar_docs = similarity_search(embedding=embedding, embedding_key='embedding', text_key='text', k=5)
        logger.info(f"Found similar documents: {len(similar_docs)}")

        # Generate the final prompt to call the LLM
        prompt = get_feedback_prompt(row, similar_docs)
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
            return jsonify({'statusCode': 200, 'body': {'response': parsed_response}})
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e}")
            return error_response(500, "Failed to parse LLM response.", raw_response=response)

    except Exception as e:
        logger.error(f"Error generating answers: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)