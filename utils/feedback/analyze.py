import os
import pymongo
from dotenv import load_dotenv
import boto3
import json
import pandas as pd
import numpy as np
from textblob import TextBlob
from rapidfuzz import fuzz, process
from utils.bedrock_call import get_prompt
from utils.docDB import get_conversation_history, similarity_search
from utils.bedrock_call import call_llm, create_embeddings
from sklearn.feature_extraction.text import CountVectorizer
load_dotenv()


def get_unique_users(df):
    unique_users = df['conv_id'].nunique()
    print(f"Total unique users: {unique_users}")
    most_frequent_users = df['conv_id'].value_counts().head(5)
    # print(most_frequent_users)
    return unique_users, most_frequent_users


def get_unique_queries(df):
    unique_questions = df['query'].nunique()
    print(f"Total unique questions: {unique_questions}")
    most_frequent_questions = df['query'].value_counts().head(5)
    # print(most_frequent_questions)
    return unique_questions, most_frequent_questions


def analyze_conversation_flow(df):
    conv_turns = df.groupby('conv_id').size()   # Calculate message count per conversation
    avg_turns = round(conv_turns.mean())
    # print(f"Average turns per conversation: {avg_turns}")
    return conv_turns, avg_turns


def group_similar_questions(df, threshold=80):
    queries = df['rewritten_query'].unique().tolist()
    grouped_queries = {}
    # Iterate through each query and find similar ones
    for query in queries:
        if query not in grouped_queries:
            matches = process.extract(query, queries, scorer=fuzz.token_set_ratio, limit=len(queries))
            similar_queries = [match[0] for match in matches if match[1] >= threshold]
            for similar_query in similar_queries:
                grouped_queries[similar_query] = query  # Assign the first query as the representative
    df['grouped_query'] = df['rewritten_query'].map(grouped_queries)
    return df

    
def analyse_user_sentiment(df):
    df['sentiment'] = df['query'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # print(df[['query', 'sentiment']].head(5))
    return df['sentiment']


def calculate_failure_rate(df):
    failed_queries = df[df['response'].isnull()].shape[0]
    total_queries = df.shape[0]
    failure_rate = failed_queries / total_queries
    return failure_rate


def get_response_accuracy(df):
    def calculate_accuracy(row):
        if pd.isna(row['response']) and not pd.isna(row['ground_truth_response']):
            return 0
        elif isinstance(row['response'], str) and row['response'] in row['ground_truth_response']:
            return 1
        elif isinstance(row['response'], str) and row['response'] and row['ground_truth_response'] and \
             any(word in row['ground_truth_response'] for word in row['response'].split()):
            return 0.5
        else:
            return 0
    df['response_accuracy'] = df.apply(calculate_accuracy, axis=1)
    accuracy = df['response_accuracy'].mean()
    return df['response_accuracy'], accuracy


def get_rewrite_accuracy(df):
    # Assuming you have a column 'correct_intent' with labeled intents
    df['intent_accuracy'] = df.apply(lambda x: 1 if x['query'] in x['rewritten_query'] else 0, axis=1)
    intent_accuracy = df['intent_accuracy'].mean()
    print(f"Intent recognition accuracy: {intent_accuracy}")
    return intent_accuracy


def analyze_session(df):
    df['time'] = pd.to_datetime(df['time'])
    session_length = df.groupby('conv_id')['time'].apply(lambda x: x.max() - x.min())
    # print(f"Session length (time difference between first and last message):{session_length}")
    return session_length


def extract_top_keywords(df):
    vectorizer = CountVectorizer(max_features=10, stop_words='english')
    X = vectorizer.fit_transform(df['query'])
    keywords = vectorizer.get_feature_names_out()
    # print(f"Top keywords: {keywords}")
    return keywords

    
def create_msg_dataset(df):
    # Ensure msg_id is an integer
    df['msg_id'] = pd.to_numeric(df['msg_id'], errors='coerce')

    # Separate user and assistant messages
    user_messages = df[df['role'] == 'user'][['conv_id', 'msg_id', 'time', 'query', 'rewritten_query']]
    assistant_responses = df[df['role'] == 'assistant'][['conv_id', 'msg_id', 'response', 'metadata', 'reference']].copy()
    
    # Align assistant msg_id to corresponding user msg_id (response comes after user)
    assistant_responses['msg_id'] = assistant_responses['msg_id'] - 1

    # Merge on conv_id and msg_id
    custom_df = pd.merge(user_messages, assistant_responses, on=['conv_id', 'msg_id'], how='left', suffixes=('_user', '_assistant'))
    custom_df = custom_df.sort_values(by=['conv_id', 'msg_id']).reset_index(drop=True)
    return custom_df


def clean_raw_response(client, raw_response):
    try:
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith('"""') and cleaned_response.endswith('"""'):
            cleaned_response = cleaned_response.strip('"""')
            response = json.loads(cleaned_response)
        return response
    except Exception as e:
        prompt = f"You need to correct the JSON format of this raw_response to be parsed successfully. raw_response:{raw_response}"
        response = call_llm(client, prompt)
        return response


def create_ground_truth(unique_questions):
    ground_truth_responses = []  # To store the ground truth responses
    ground_truth_metadata = []  # To store metadata
    ground_truth_references = []  # To store references
    
    client = pymongo.MongoClient(os.getenv("CONNECTION_STRING"))

    for rewritten_query in unique_questions:
        try:
            client = boto3.client(service_name='bedrock-runtime', region_name="ap-south-1")
            embed_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")
            print("Creating embeddings for the query.")
            query_embedding = create_embeddings(embed_client, rewritten_query)

            if query_embedding is None:
                print("Error: Failed to create embeddings for the query.")
                ground_truth_responses.append(None)
                ground_truth_metadata.append(None)
                ground_truth_references.append(None)
                continue  # Skip this iteration

            # Search for similar documents in the database
            similar_docs = similarity_search(embedding=query_embedding, embedding_key='embedding', text_key='text', k=5)

            # Generate the final prompt to call the LLM
            prompt = get_prompt(rewritten_query, similar_docs, None)

            # Call LLM to get the final response using the 70B model
            response = call_llm(client, prompt, model='meta.llama3-70b-instruct-v1:0')

            if response is None:
                print("Error: Failed to generate a response from LLM.")
                ground_truth_responses.append(None)
                ground_truth_metadata.append(None)
                ground_truth_references.append(None)
                continue  # Skip this iteration

            # Parse the response and validate the structure
            try:
                parsed_response = json.loads(response)
                print("LLM response parsed successfully.")
            except json.JSONDecodeError:
                print("Error: JSONDecodeError occurred. Attempting to clean response.")
                result = clean_raw_response(client, response)
                try:
                    parsed_response = json.loads(result)
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse LLM response: {e}")
                    ground_truth_responses.append(None)
                    ground_truth_metadata.append(None)
                    ground_truth_references.append(None)
                    continue  # Skip this iteration

            # Append parsed response data
            ground_truth_responses.append(parsed_response.get('Answer'))
            ground_truth_metadata.append(parsed_response.get('Metadata'))
            ground_truth_references.append(parsed_response.get('References'))

        except Exception as e:
            print(f"Error: Unexpected error in create_ground_truth: {e}")
            ground_truth_responses.append(None)
            ground_truth_metadata.append(None)
            ground_truth_references.append(None)

    # Create a DataFrame to hold the ground truth responses for unique questions
    results_df = pd.DataFrame({
        'rewritten_query': unique_questions,
        'ground_truth_response': ground_truth_responses,
        'ground_truth_metadata': ground_truth_metadata,
        'ground_truth_reference': ground_truth_references
    })

    return results_df
    

def get_eval_prompt(row):
    prompt = f"""
Evaluate the following response based on these metrics: faithfulness, context precision, answer correctness, answer relevancy, and context recall.

**Question:** {row['rewritten_query']}
**Response:** {row['response']}
**Ground Truth:** {row['ground_truth_response']}
**Contexts:** {row['ground_truth_reference'] if row['ground_truth_reference'] else "No context available."}

Please provide scores for each metric in JSON format as shown below:
{{
    "Faithfulness": "<Score between 0 and 1>",
    "Context_precision": "<Score between 0 and 1>",
    "Answer_correctness": "<Score between 0 and 1>",
    "Answer_relevancy": "<Score between 0 and 1>",
    "Context_recall": "<Score between 0 and 1>",
    "Reason": "<Brief explanation (max 20 words)>"
}}

Return only the output JSON without any additional text.
"""
    return prompt


def get_evaluation_metrics(df):
    df['response'].fillna("", inplace=True)  # Avoid chaining warnings
    scores = { 
        'faithfulness': [], 
        'context_precision': [], 
        'answer_correctness': [], 
        'answer_relevancy': [], 
        'context_recall': [], 
        'reason': []
    }
    for index, row in df.iterrows():
        client = boto3.client(service_name='bedrock-runtime', region_name="ap-south-1")
        eval_prompt = get_eval_prompt(row)
        scores_response = call_llm(client, eval_prompt, model='meta.llama3-70b-instruct-v1:0')
        print(scores_response)
        # Handle possible JSON response
        try:
            evaluation_scores = json.loads(scores_response)
            print(evaluation_scores)
            for key in scores.keys():
                print(key)
                scores[key].append(evaluation_scores.get(key.capitalize(), None))
                print(evaluation_scores.get(key.capitalize()))
        except json.JSONDecodeError:
            print(f"Error: Failed to parse evaluation scores for index {index}")
    for key, value in scores.items():
        df[key] = value
    return df


def get_ragas_analysis(df=None, file_name=None, top_n=None, save_file="analysis_report.xlsx"):
    if file_name:
        df = pd.read_excel(file_name)
    # Grouped questions
    df = df[df['response'].notna() & (df['response'] != "")]
    df = group_similar_questions(df)
    
    # Count top most asked unique grouped queries
    if top_n:
        top_questions = df['grouped_query'].value_counts().head(top_n).index.tolist()
    else:
        top_questions = df['grouped_query'].value_counts().index.tolist()
    print(f"Creting Ground Truth of top {len(top_questions)} questions")
    ground_truth_df = create_ground_truth(top_questions)

    # Filter and merge
    custom_df = df[df['grouped_query'].isin(top_questions)].copy()
    custom_df = custom_df.merge(ground_truth_df, on='rewritten_query', how='left')
    custom_df = custom_df.drop_duplicates(subset='grouped_query', keep='first')
    custom_df = custom_df.drop(columns=['grouped_query'])

    # Calculate additional metrics
    failure_rate = calculate_failure_rate(df)
    rewrite_accuracy = get_rewrite_accuracy(df)
    custom_df['response_accuracy'], avg_accuracy = get_response_accuracy(custom_df)

    custom_df = get_evaluation_metrics(custom_df)
    
    if not os.path.exists(save_file):
        with pd.ExcelWriter(save_file, engine='openpyxl', mode='w') as writer:
            custom_df.to_excel(writer, sheet_name='Data w Ground Truth and analysis', index=False)
    else:
        with pd.ExcelWriter(save_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            custom_df.to_excel(writer, sheet_name='Data w Ground Truth and analysis', index=False)
    with pd.ExcelWriter(save_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        # Save failure rate and accuracy
        summary = pd.DataFrame({
            'Metric': ['Failure Rate', 'Rewrite Accuracy', 'Response Accuracy'],
            'Value': [failure_rate, rewrite_accuracy, avg_accuracy]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    print(f"Ragas Analysis results with ground truth are successfully saved to {save_file}")


def analyze_history(df=None, file_name=None, save_file="analysis_report.xlsx"):
    if file_name:
        df = pd.read_excel(file_name)

    # Perform analysis
    unique_users, most_frequent_users = get_unique_users(df)
    unique_questions, most_frequent_questions = get_unique_queries(df)
    conv_size, avg_conv_size = analyze_conversation_flow(df)
    df['sentiment'] = analyse_user_sentiment(df)
    session_length = analyze_session(df)
    most_inquired_topics = extract_top_keywords(df)

    # Check if the file already exists and Save analysis results to an Excel file
    if not os.path.exists(save_file):
        with pd.ExcelWriter(save_file, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name='Data w sentiment analysis', index=False)
    else:
        with pd.ExcelWriter(save_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='Data w sentiment analysis', index=False)
    with pd.ExcelWriter(save_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        user_data = most_frequent_users.reset_index().rename(columns={'conv_id': 'User', 'count': 'Count'})
        user_data.to_excel(writer, sheet_name='User Analysis', index=False)

    data = []
    data.append('Value','Stats')
    data.append('Unique Users', unique_users)
    data.append('Unique Questions', unique_questions)
    data.append('Average Conversation Size', avg_conv_size)

    with pd.ExcelWriter(save_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        # Save user analysis
        data.to_excel(writer, sheet_name='General Analysis', index=False)
        
        # Save query analysis
        query_data = most_frequent_questions.reset_index().rename(columns={'query': 'Query', 'count': 'Count'})
        query_data.to_excel(writer, sheet_name='Query Analysis', index=False)
        
        # Save conversation flow analysis
        combined_data = pd.DataFrame({
            'conv_id': conv_size.index,
            'Interaction Count': conv_size.values,
            'Session Length': session_length.values
        })
        combined_data.to_excel(writer, sheet_name='Conversation Analysis', index=False)

        # Save keywords
        keywords_data = pd.DataFrame(most_inquired_topics, columns=['Keyword'])
        keywords_data.to_excel(writer, sheet_name='Top Keywords', index=False)

    print(f"Analysis successfully saved to {save_file}")
    return df
    

def update_conversation_history(file_name='conversation_data.xlsx', save_file='conv_history.xlsx'):
    try:
        print("Fetching Data from documentDB")
        df = get_conversation_history()
        print(f"Fetched Data successfully from documentDB. Found {len(df)} conversations")
    except Exception as e:
        print(f"Failed to fetch responses")
        return e

    # df = pd.read_excel(file_name)
    if df is not None and len(df) > 0:
        df.to_excel(file_name, index=False)
        print(f"Data successfully saved to {file_name}")
        
        conv = create_msg_dataset(df)
        conv.to_excel(save_file, index=False)
        print(f"Data successfully saved to {save_file}")
    else:
        print("No data to save.")

if __name__ == '__main__':
    file_name = 'conv_history.xlsx'
    update_conversation_history(save_file=file_name)
    
    df = analyze_history(file_name=file_name)
    if df.empty:
        get_ragas_analysis(file_name)
    else:
        get_ragas_analysis(df)