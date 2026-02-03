import requests
import json
import schedule
import time
import logging
import pandas as pd

# Set up logging for both success and error messages
logging.basicConfig(
    filename='fetch_and_save.log',  # Log file name
    level=logging.INFO,             # Log level (INFO captures everything from info level and above)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

# Scheduling control variables
initial_interval = 1  # Start with a 1-minute interval
max_interval = 1000  # Cap at 400 minutes (or any desired max)
current_interval = initial_interval  # Current interval starts at the initial value
current_row_index = 0  # Start processing from the first row

# Separate logger for errors (to also log errors in the main log file)
error_logger = logging.getLogger('error_logger')
error_file_handler = logging.FileHandler('fetch_and_save_errors.log')
error_file_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_file_handler)

def fetch_and_save():
    global current_interval, current_row_index
    try:
        # Load the Excel file
        df = pd.read_excel('conv_history.xlsx')#Updated_HR_bot_sheet_previous.xlsx')

        # Check if the current row index is within the DataFrame bounds
        if current_row_index >= len(df):
            logging.info("All rows processed. Restarting from the first row.")
            current_row_index = 0  # Reset to the beginning once all rows are processed

        # Process only one row per cycle
        row = df.iloc[current_row_index]
        query = row['question']
        data = {"query": query, "conversation_id": "123456"}
        logging.info(f"Processing query {current_row_index}: {query}")

        try:
            # Define the URL and headers
            # url = 'https://tqrzptekwjdlqleb43otv3qmu40svnad.lambda-url.ap-south-1.on.aws/'
            url = 'http://52.66.10.81:5004/process_query'
            headers = {'Content-Type': 'application/json'}

            # Send the request
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Check for HTTP errors
            response_json = response.json()
            response_text = response_json.get('response', {}).get('Answer', 'No response')
            
            # Log the response text
            logging.info(f"Query {current_row_index+1} Answer: {response}")
            df.at[current_row_index, 'answer'] = response_text
        except requests.exceptions.RequestException as req_err:
            error_message = f"Request error for query {current_row_index+1}: {req_err}"
            logging.error(error_message)
            error_logger.error(error_message)  # Log in both main and error-specific log files
        except json.JSONDecodeError as json_err:
            error_message = f"JSON decode error for query {current_row_index+1}: {json_err}"
            logging.error(error_message)
            error_logger.error(error_message)
        except KeyError as key_err:
            error_message = f"Key error for query {current_row_index+1}: {key_err}"
            logging.error(error_message)
            error_logger.error(error_message)
        except Exception as e:
            error_message = f"Unexpected error for query {current_row_index+1}: {e}"
            logging.error(error_message)
            error_logger.error(error_message)

        # Save the updated Excel sheet after processing each row
        df.to_excel('Updated_HR_bot_sheet_previous_ans.xlsx', index=False)
        logging.info(f"Updated Excel file saved successfully.")

        # Update the row index for the next run
        current_row_index += 1

    except Exception as e:
        error_message = f"Failed to read Excel file or process query at index {current_row_index+1}: {e}"
        logging.error(error_message)
        error_logger.error(error_message)

    # Exponential backoff scheduling logic
    current_interval = min(round(current_interval * 1.5), max_interval)
    schedule.clear()  # Clear existing schedules
    schedule.every(current_interval).minutes.do(fetch_and_save)  # Reschedule with new interval
    logging.info(f"Next run scheduled in {current_interval} minutes.")

# Initial schedule setup
schedule.every(current_interval).minutes.do(fetch_and_save)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)