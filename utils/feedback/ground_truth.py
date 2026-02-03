import pandas as pd
from langchain_aws import BedrockLLM
from utils.docDB import similarity_search
from bedrock_call import get_prompt

# Initialize the 70B Llama model from Bedrock
ground_truth_llm = BedrockLLM(
    model_id="meta.llama3-1-70b-instruct-v1:0", region_name='ap-south-1', streaming=True
)
embed_model = BedrockLLM(
    model_id="amazon.titan-embed-text-v2:0", region_name='us-west-2', streaming=True
)

# Load your data (assuming it's in a CSV or Excel format)
data = pd.read_excel("conv_history.xlsx")  # Replace with the path to your dataset

# Function to generate responses from the Llama 70B model
def generate_ground_truth(query):
    
    query_embedding = embed_model.generate([query])
    similar_docs = similarity_search(embedding=query_embedding, embedding_key='embedding', text_key='text', k=5)
    logger.info(f"Found similar documents: {len(similar_docs)}")

    # Generate the final prompt to call the LLM
    prompt = get_prompt(rewritten_query, similar_docs, conv_id)
    logger.info("Prompt generated successfully.")

    # Call LLM to get the final response
    logger.info("Calling LLM to generate the response.")
    return ground_truth_llm.generate(prompt)

# Iterate over each row in the DataFrame, generating the ground truth
data['ground_truth'] = data['rewritten_query'].apply(generate_ground_truth)

# Save the updated dataset with ground truth to a new file
data.to_csv("data_with_ground_truth.csv", index=False)

print("Ground truth generation complete. Data saved to 'data_with_ground_truth.csv'")