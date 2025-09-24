import google.generativeai as genai
import asyncio
import os
import pandas as pd
import random

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Set random seed for reproducibility
random.seed(123)

def load_data(csv_path):
    """Load subreddit rules data from CSV file"""
    try:
        # Load the CSV with Subreddit, Rule Name, and Rule Description columns
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_columns = ['Subreddit', 'Rule Name', 'Rule Description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Loaded {len(df)} rules from {df['Subreddit'].nunique()} subreddits")
        return df
        
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        raise
    except Exception as e:
        print(f"Error processing CSV: {e}")
        raise

def create_rule_formatting_prompt(subreddit, rule_name, rule_description):
    """Create prompt to format rule according to specifications"""
    return f"""You are formatting Reddit moderation rules for /r/{subreddit}.

Given this rule:
Original Rule Name: {rule_name}
Original Rule Description: {rule_description}

Your task is to format this rule following these exact specifications:

1. **Formatted Rule Name**: Create a concise rule name (1-5 words) that captures the essence
2. **Formatted Rule Description**: Create a clear description (10-20 words) that explains what the rule prohibits/requires
3. **Clarity Assessment**: Determine if the original rule was clear and unambiguous

Format your response EXACTLY like this:
Formatted Rule Name: [1-5 words]
Formatted Rule Description: [10-20 words describing what is prohibited/required]
Final Formatted Rule: [Rule Name]: [Rule Description]
Clarity Tag: Clear OR Unclear
Clarity Reason: [Brief explanation if marked as Unclear, otherwise "Rule is clear and well-defined"]

Example format:
Formatted Rule Name: No Advertising
Formatted Rule Description: Spam, referral links, unsolicited advertising, and promotional content are not allowed
Final Formatted Rule: No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed
Clarity Tag: Clear
Clarity Reason: Rule is clear and well-defined"""

def parse_formatting_response(response_text):
    """Parse the generated response into structured data"""
    if not response_text:
        return {}
    
    parsed_data = {}
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Formatted Rule Name:'):
            parsed_data['formatted_rule_name'] = line.replace('Formatted Rule Name:', '').strip()
        elif line.startswith('Formatted Rule Description:'):
            parsed_data['formatted_rule_description'] = line.replace('Formatted Rule Description:', '').strip()
        elif line.startswith('Final Formatted Rule:'):
            parsed_data['final_formatted_rule'] = line.replace('Final Formatted Rule:', '').strip()
        elif line.startswith('Clarity Tag:'):
            parsed_data['clarity_tag'] = line.replace('Clarity Tag:', '').strip()
        elif line.startswith('Clarity Reason:'):
            parsed_data['clarity_reason'] = line.replace('Clarity Reason:', '').strip()
    
    return parsed_data

async def process_single_rule(subreddit, rule_name, rule_description):
    """Process a single rule - format it according to specifications"""
    try:
        prompt = create_rule_formatting_prompt(subreddit, rule_name, rule_description)
        
        response = await model.generate_content_async(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Lower temperature for more consistent formatting
                top_p=0.9
            )
        )
        
        if response.candidates and response.candidates[0].content.parts:
            raw_response = response.candidates[0].content.parts[0].text
            parsed_data = parse_formatting_response(raw_response)
            
            return {
                'Subreddit': subreddit,
                'Original_Rule_Name': rule_name,
                'Original_Rule_Description': rule_description,
                'Formatted_Rule_Name': parsed_data.get('formatted_rule_name'),
                'Formatted_Rule_Description': parsed_data.get('formatted_rule_description'),
                'Final_Formatted_Rule': parsed_data.get('final_formatted_rule'),
                'Clarity_Tag': parsed_data.get('clarity_tag'),
                'Clarity_Reason': parsed_data.get('clarity_reason'),
                'Raw_Response': raw_response,
                'Error': None
            }
        else:
            return {
                'Subreddit': subreddit,
                'Original_Rule_Name': rule_name,
                'Original_Rule_Description': rule_description,
                'Formatted_Rule_Name': None,
                'Formatted_Rule_Description': None,
                'Final_Formatted_Rule': None,
                'Clarity_Tag': None,
                'Clarity_Reason': None,
                'Raw_Response': None,
                'Error': 'No response from model'
            }
            
    except Exception as e:
        return {
            'Subreddit': subreddit,
            'Original_Rule_Name': rule_name,
            'Original_Rule_Description': rule_description,
            'Formatted_Rule_Name': None,
            'Formatted_Rule_Description': None,
            'Final_Formatted_Rule': None,
            'Clarity_Tag': None,
            'Clarity_Reason': None,
            'Raw_Response': None,
            'Error': str(e)
        }

async def process_batch_async(batch_data):
    """Process a batch of rules"""
    tasks = []
    for subreddit, rule_name, rule_description in batch_data:
        tasks.append(process_single_rule(subreddit, rule_name, rule_description))
    
    print(f"Processing {len(tasks)} rules...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("Batch completed.")
    
    # Handle any exceptions
    clean_results = []
    for result in results:
        if isinstance(result, dict):
            clean_results.append(result)
        else:
            print(f"Error in processing: {result}")
    
    return clean_results

async def main():
    """Main function to process and format rules"""
    # Load input data
    input_csv_path = '/home/vino/ML_Projects/Jigsaw-ACRC-Kaggle/data/synthetic_generation/popular_subreddit_rules_kaggle_subreddit.csv'  # Update this path to your CSV file
    
    print("Loading subreddit rules data...")
    rules_df = load_data(input_csv_path)
    #rules_df = rules_df.iloc[12000:]

    print(f"Processing {len(rules_df)} rules from {rules_df['Subreddit'].nunique()} subreddits")
    
    # Show sample of input data
    print(f"\nSample input data:")
    print(rules_df.head(3)[['Subreddit', 'Rule Name', 'Rule Description']])
    
    # Prepare data for processing
    rule_combinations = []
    for _, row in rules_df.iterrows():
        rule_combinations.append((row['Subreddit'], row['Rule Name'], row['Rule Description']))
    
    print(f"\nFormatting {len(rule_combinations)} rules...")
    
    all_results = []
    batch_size = 100
    
    # Process in batches
    for i in range(0, len(rule_combinations), batch_size):
        batch = rule_combinations[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"\n--- Processing batch {batch_num} ({len(batch)} rules) ---")
        
        results = await process_batch_async(batch)
        all_results.extend(results)

        # Save batch results
        if results:
            batch_df = pd.DataFrame(results)
            filename = f"formatted_rules_batch_{batch_num}.csv"
            batch_df.to_csv(filename, index=False)
            print(f"Saved batch to {filename}")

    # Save all results
    if all_results:
        final_df = pd.DataFrame(all_results)
        output_path = 'formatted_subreddit_rules_kaggle_data.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved all formatted rules to {output_path}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total rules processed: {len(final_df)}")
        print(f"Successfully formatted: {final_df['Final_Formatted_Rule'].notna().sum()}")
        print(f"Success rate: {(final_df['Final_Formatted_Rule'].notna().sum() / len(final_df)) * 100:.1f}%")
        print(f"Clear rules: {(final_df['Clarity_Tag'] == 'Clear').sum()}")
        print(f"Unclear rules: {(final_df['Clarity_Tag'] == 'Unclear').sum()}")
        print(f"Unique subreddits: {final_df['Subreddit'].nunique()}")
        
        # Show sample formatted rules
        print(f"\nSample formatted rules:")
        sample_df = final_df[final_df['Final_Formatted_Rule'].notna()].head(5)
        for _, row in sample_df.iterrows():
            print(f"\n/r/{row['Subreddit']}:")
            print(f"  Original: {row['Original_Rule_Name']} - {row['Original_Rule_Description']}")
            print(f"  Formatted: {row['Final_Formatted_Rule']}")
            print(f"  Clarity: {row['Clarity_Tag']}")
        
        # Show unclear rules if any
        unclear_rules = final_df[final_df['Clarity_Tag'] == 'Unclear']
        if len(unclear_rules) > 0:
            print(f"\nSample unclear rules:")
            for _, row in unclear_rules.head(3).iterrows():
                print(f"\n/r/{row['Subreddit']}: {row['Final_Formatted_Rule']}")
                print(f"  Reason: {row['Clarity_Reason']}")
        
        print(f"\nOutput columns: {list(final_df.columns)}")
        print(f"Output shape: {final_df.shape}")
        
        return final_df
    else:
        print("No results generated")
        return None

if __name__ == "__main__":
    result_df = asyncio.run(main())