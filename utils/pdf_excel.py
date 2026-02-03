import uuid
import logging
import pandas as pd
from io import BytesIO
from utils.bucket_call import extract_file_from_s3
from utils.bedrock_call import create_embeddings
from utils.text_extraction import extract_text_from_pdf
from utils.docDB import insert_many_entry
from utils.chunk_embed import create_chunks

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def process_pdf(client, bucket_name, object_key, eventSource, eventTime):
    extracted_text = extract_text_from_pdf(bucket_name, object_key)
    if extracted_text is None:
        raise ValueError("Failed to extract text from the document.")

    chunk_infos = []
    for page, text in extracted_text.items():
        logger.info(f"Processing Page {page}")
        chunks = create_chunks(text)

        for i, chunk in enumerate(chunks):
            embedding = create_embeddings(client, chunk)
            if embedding is not None:
                chunk_info = {
                    'text': chunk,
                    'embedding': embedding,
                    'upload_source': eventSource,
                    'uuid': str(uuid.uuid1()),
                    'datetime': eventTime,
                    'doc_type': "pdf",
                    'metadata': {
                        'file_name': object_key,
                        'page_no': page,
                        'chunk_no': i
                    }
                }
                chunk_infos.append(chunk_info)

    ids = insert_many_entry(chunk_infos)
    logger.info(f"Documents inserted successfully: {ids}")

def process_excel_csv(client, bucket_name, object_key, eventSource, eventTime):
    s3_response = extract_file_from_s3(bucket_name, object_key)
    file_content = s3_response['Body'].read()
    logger.info(f'{object_key} data fetched from S3 bucket: {bucket_name}')

    if object_key.endswith('.xlsx'):
        data = pd.read_excel(BytesIO(file_content), sheet_name=None, header=None)
        cleaned_excel_data = clean_excel_data(data)
    elif object_key.endswith('.csv'):
        try:
            data = pd.read_csv(BytesIO(file_content), encoding='utf-8', header=None)
        except UnicodeDecodeError:
            data = pd.read_csv(BytesIO(file_content), encoding='latin1', header=None)
        cleaned_excel_data = clean_excel_data({'Sheet1': data}) # Wrap CSV data in a dictionary to handle it uniformly
    else:
        raise ValueError("Unsupported file format")

    if cleaned_excel_data:
        logger.info(f"Data cleaning done, proceeding to create embeddings")
        docs = create_docs_from_data(client, cleaned_excel_data, eventSource, eventTime, object_key)
        ids = insert_many_entry(docs)
        logger.info(f"Data processed and stored successfully: {ids}")

def clean_excel_data(data):
    cleaned_data = {}
    for sheet_name, sheet_data in data.items():
        # Check if sheet_data is a DataFrame
        if isinstance(sheet_data, pd.DataFrame):
            if 0 in sheet_data.columns and 1 in sheet_data.columns:
                initial_row_count = len(sheet_data)               
                sheet_data = sheet_data.dropna(subset=[0, 1], how='all') # Drop rows where both columns 'A' and 'B' (index 0 and 1) are NaN
                cleaned_row_count = len(sheet_data)
                if cleaned_row_count < initial_row_count:
                    dropped_row_percentage = ((initial_row_count - cleaned_row_count) / initial_row_count) * 100
                    logger.info(f"Sheet {sheet_name}: Dropped {initial_row_count - cleaned_row_count} rows ({dropped_row_percentage:.2f}%) due to null values.")
                cleaned_data[sheet_name] = sheet_data
    return cleaned_data

def create_docs_from_data(client, data, eventSource, eventTime, object_key):
    docs = []
    for sheet_name, sheet_data in data.items():
        for index, row in sheet_data.iterrows():
            question = row[0]
            answer = row[1]

            if pd.notna(question) and pd.notna(answer):
                combined_text = f"Question: {question} Answer: {answer}"
                combined_embedding = create_embeddings(client, combined_text)

                if combined_embedding is None:
                    continue

                doc = {
                    'text': combined_text,
                    'embedding': combined_embedding,
                    'upload_source': eventSource,
                    'uuid': str(uuid.uuid1()),
                    'datetime': eventTime,
                    'doc_type': "xlsx" if object_key.endswith('.xlsx') else "csv",
                    'metadata': {
                        'file_name': object_key,
                        'sheet_name': sheet_name,
                        'index': index
                    }
                }
                docs.append(doc)
                logger.info("Sheet index added: ", sheet_name, "-", index)
    return docs