import google.generativeai as genai
import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score
import re

load_dotenv()

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")

def create_prompt(input_data: pd.Series):
    """Create prompt for Gemini API based on input data"""
    # Create system message
    system_msg = (
        f"You are an expert moderator for /r/{input_data['subreddit']}. "
        "Your task is to determine if a comment violates the rule. "
        "Answer strictly with 'Yes' or 'No'."
    )

    # User message: structured, concise, and clearly separated examples
    user_msg = (f"""Rule: {input_data['rule']}
        Example 1:
        {input_data['positive_example_1']}
        Rule violation: Yes
        Example 2:
        {input_data['negative_example_1']}
        Rule violation: No
        Example 3:
        {input_data['positive_example_2']}
        Rule violation: Yes
        Example 4:
        {input_data['negative_example_2']}
        Rule violation: No
        Test sentence:
        {input_data['body']}""")

    return f"{system_msg}\n\n{user_msg}"

def standardize_response(response):
    """Standardize response to only 'Yes' or 'No'"""
    if pd.isna(response) or response == "" or response is None:
        return None

    # Convert to string and clean
    response_str = str(response).strip()

    # Use regex to find Yes/No patterns (case insensitive)
    yes_pattern = re.compile(r'\b(yes|y)\b', re.IGNORECASE)
    no_pattern = re.compile(r'\b(no|n)\b', re.IGNORECASE)

    # Check for Yes first, then No
    if yes_pattern.search(response_str):
        return "Yes"
    elif no_pattern.search(response_str):
        return "No"
    else:
        # If neither Yes nor No is found, return None for manual review
        return None

async def get_gemini_response(input_data: pd.Series, index: int):
    """Get response from Gemini API for a single row"""
    try:
        prompt = create_prompt(input_data)
        response = await model.generate_content_async(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )

        if response.candidates and response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text.strip()
            return {
                'index': index,
                'response': response_text,
                'error': None
            }
        else:
            return {
                'index': index,
                'response': None,
                'error': 'No response from API'
            }

    except Exception as e:
        return {
            'index': index,
            'response': None,
            'error': str(e)
        }

async def process_batch_async(df_batch, start_index):
    """Process a batch of DataFrame rows asynchronously"""
    tasks = []
    for i, (idx, row) in enumerate(df_batch.iterrows()):
        tasks.append(get_gemini_response(row, start_index + i))

    print(f"Processing batch starting at index {start_index} with {len(tasks)} items...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"Batch completed.")

    return results

async def process_dataframe_async(df, batch_size=50, save_path=None):
    """Process entire DataFrame asynchronously with batching"""
    print(f"Processing DataFrame with {len(df)} rows...")

    responses = [None] * len(df)
    errors = [None] * len(df)

    # Process in batches
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"\n--- Processing batch {batch_num} ({i+1}-{min(i+batch_size, len(df))}) ---")

        results = await process_batch_async(batch_df, i)

        # Store results in order
        for result in results:
            if isinstance(result, Exception):
                print(f"Exception in batch: {result}")
                continue

            idx = result['index']
            responses[idx] = result['response']
            errors[idx] = result['error']

        # Save intermediate results if path provided
        if save_path:
            temp_df = df.copy()
            temp_df['gemini_response'] = responses
            temp_df['error'] = errors
            temp_df.to_csv(f"{save_path}_batch_{batch_num}.csv", index=False)
            print(f"Saved batch {batch_num} to {save_path}_batch_{batch_num}.csv")

    # Add responses to original DataFrame
    df_result = df.copy()
    df_result['gemini_response'] = responses
    df_result['error'] = errors

    # Save final results
    if save_path:
        df_result.to_csv(f"{save_path}_final.csv", index=False)
        print(f"\nSaved final results to {save_path}_final.csv")

    # Print summary
    success_count = sum(1 for r in responses if r is not None)
    error_count = sum(1 for e in errors if e is not None)
    print(f"\nProcessing completed:")
    print(f"Success: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
    print(f"Errors: {error_count}/{len(df)} ({error_count/len(df)*100:.1f}%)")

    return df_result

# Main function to run the processing
async def main(df_train_0, save_path="gemini_responses"):
    """Main function to process df_train_0 with Gemini API"""
    result_df = await process_dataframe_async(df_train_0, batch_size=100, save_path=save_path)
    return result_df

# Synchronous wrapper function
def process_with_gemini(df_train_0, save_path="gemini_responses"):
    """Synchronous wrapper to process DataFrame with Gemini API"""
    return asyncio.run(main(df_train_0, save_path))

#Example usage:
if __name__ == "__main__":
    # Load your df_train_0 here
    df_train_0 = pd.read_csv('/home/vino/ML_Projects/Jigsaw-ACRC-Kaggle/data/final/df_train_0.csv')
    df_train_0 = df_train_0.sample(n=300, random_state=42)

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    result = process_with_gemini(df_train_0, "output/gemini_responses")
    print(result.head())

    # Calculate F1 score between violates_rule and gemini prediction
    # Filter out rows with no gemini response
    valid_responses = result[result["gemini_response"].notna() & (result["gemini_response"] != "")]

    print(f"Valid responses for scoring: {len(valid_responses)}/{len(result)} ({len(valid_responses)/len(result)*100:.1f}%)")

    if len(valid_responses) > 0:
        # Standardize both violates_rule and gemini_response to Yes/No
        valid_responses = valid_responses.copy()
        valid_responses["violates_rule"] = valid_responses["violates_rule"].map({True: "Yes", False: "No", "Yes": "Yes", "No": "No"})
        valid_responses["gemini_response_standardized"] = valid_responses["gemini_response"].apply(standardize_response)

        # Filter out rows where standardization failed
        standardized_responses = valid_responses[valid_responses["gemini_response_standardized"].notna()]

        print(f"Successfully standardized responses: {len(standardized_responses)}/{len(valid_responses)} ({len(standardized_responses)/len(valid_responses)*100:.1f}%)")

        if len(standardized_responses) > 0:
            # Compute F1 score (convert Yes/No to 1/0)
            y_true = standardized_responses["violates_rule"].map({"Yes": 1, "No": 0})
            y_pred = standardized_responses["gemini_response_standardized"].map({"Yes": 1, "No": 0})

            # Remove any remaining NaN values
            valid_mask = y_true.notna() & y_pred.notna()
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]

            if len(y_true) > 0:
                f1 = f1_score(y_true, y_pred)
                print("F1-score:", f1)
            else:
                print("No valid predictions for F1 score calculation")
        else:
            print("No standardized responses available for F1 score calculation")
    else:
        print("No valid responses available for scoring")
