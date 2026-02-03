import time
import json
import pandas as pd
import requests
from datetime import datetime
from secrets_manager import API_KEY, FLASK_API_IP

FLASK_API_URL = f"{FLASK_API_IP}/process_query"
EXCEL_FILE_PATH = 'Data/policybot_questions.xlsx'

def read_questions_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        questions = df['question'].tolist()
        return questions
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return []

def send_query_to_flask(query, conversation_id):
    headers = {
        'x-api-key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    data = {
        "query": query,
        "conversation_id": conversation_id
    }
    
    try:
        response = requests.post(FLASK_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            print(f"Successfully sent query: {query}")
            return response.json()
        else:
            print(f"Error sending query: {query}. Status Code: {response.status_code}, Message: {response.text}")
            return None
    except requests.RequestException as e:
        print(f"Error making the API request: {e}")
        return None

def append_to_excel(data, file_path):
    try:
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name='Results', index=False, header=writer.sheets.get('Results') is None)
        print("Successfully appended results to Excel.")
    except Exception as e:
        print(f"Error appending to Excel: {e}")


def run_cron_job():
    questions = read_questions_from_excel(EXCEL_FILE_PATH)
    
    if not questions:
        print("No questions found in the Excel file.")
        return
    
    conversation_id = f"conv_{int(time.time())}"
    
    for question in questions:
        print(f"Sending question: {question} (Conversation ID: {conversation_id})")
        response = send_query_to_flask(question, conversation_id)
        if response:
            result_entry = {
                'question': question,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'response': json.dumps(response)  # Store the response as a JSON string
            }
            append_to_excel(result_entry, EXCEL_FILE_PATH)
            results.append
        
        print(f"Waiting for 10 minutes before sending the next query...")
        time.sleep(600)

if __name__ == '__main__':
    run_cron_job()