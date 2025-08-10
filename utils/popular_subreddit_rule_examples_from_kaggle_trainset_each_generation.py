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
model = genai.GenerativeModel("gemini-2.5-flash")

# Set random seed for reproducibility
random.seed(149831)

def load_data(csv_path):
    """Load subreddit rules data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
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

def load_kaggle_examples(csv_path):
    """Load kaggle examples dataset"""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['subreddit', 'rule', 'positive_example_1', 'positive_example_2', 
                           'negative_example_1', 'negative_example_2']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in kaggle dataset: {missing_columns}")
        
        print(f"Loaded {len(df)} example comments from kaggle dataset")
        return df
        
    except FileNotFoundError as e:
        print(f"Error loading kaggle examples: {e}")
        raise
    except Exception as e:
        print(f"Error processing kaggle CSV: {e}")
        raise

def get_random_example_comments(kaggle_df):
    """Get random examples from kaggle dataset"""
    if len(kaggle_df) == 0:
        return ""
    
    # Randomly select a row
    random_row = kaggle_df.sample(n=1).iloc[0]
    
    # Create example_comments string with all examples
    example_comments = f"""
Example Reddit Comments from /r/{random_row['subreddit']}:

Positive Examples (Rule Violations):
- "{random_row['positive_example_1']}"
- "{random_row['positive_example_2']}"

Negative Examples (Good Comments):
- "{random_row['negative_example_1']}"
- "{random_row['negative_example_2']}"

Rule Context: {random_row['rule']}
"""
    
    return example_comments

def create_rule_processing_prompt(subreddit, rule_name, rule_description, example_comments):
    """Create prompt to format rule and generate realistic Reddit examples"""
    return f"""You are processing Reddit moderation rules for /r/{subreddit}.

Given this rule:
Rule Name: {rule_name}
Rule Description: {rule_description}

Use these REAL Reddit comment examples as inspiration for style and authenticity:
{example_comments}

Based on these real examples, perform these tasks:

1. Create a concise, clear formatted version of this rule (around 8-12 words) that captures the essence for moderation purposes.

2. Generate NEW realistic Reddit comment examples for /r/{subreddit} that follow similar patterns to the provided examples. Make them authentic with Reddit language including:
   - Internet slang and Reddit terminology
   - Casual grammar and spelling
   - Emojis when appropriate
   - Self-promotion attempts for positive examples
   - Spam-like content patterns
   - Gaming references and memes
   - Casual/crude language
   - ALL CAPS text occasionally
   - Random promotional content

Learn from the style and patterns in the provided real examples to create new, similar examples.

Format your response EXACTLY like this:
Formatted Rule: [concise 1-line rule in 10-20 words]
Positive Example 1: [NEW realistic Reddit comment that violates the rule, inspired by the real examples]
Negative Example 1: [NEW realistic Reddit comment that doesn't violate the rule]
Positive Example 2: [NEW realistic Reddit comment that violates the rule, inspired by the real examples]
Negative Example 2: [NEW realistic Reddit comment that doesn't violate the rule]
Test Comment: [NEW realistic Reddit comment for testing - randomly violates or doesn't violate the rule]
Violates Rule: True or False for the test comment

Make the comments feel authentic to /r/{subreddit} community with real Reddit chaos, slang, and behavior patterns with words from 10-50 in each example and test comment."""

def parse_generated_response(response_text):
    """Parse the generated response into structured data"""
    if not response_text:
        return {}
    
    parsed_data = {}
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Formatted Rule:'):
            parsed_data['formatted_rule'] = line.replace('Formatted Rule:', '').strip()
        elif line.startswith('Positive Example 1:'):
            parsed_data['positive_example_1'] = line.replace('Positive Example 1:', '').strip()
        elif line.startswith('Negative Example 1:'):
            parsed_data['negative_example_1'] = line.replace('Negative Example 1:', '').strip()
        elif line.startswith('Positive Example 2:'):
            parsed_data['positive_example_2'] = line.replace('Positive Example 2:', '').strip()
        elif line.startswith('Negative Example 2:'):
            parsed_data['negative_example_2'] = line.replace('Negative Example 2:', '').strip()
        elif line.startswith('Test Comment:'):
            parsed_data['test_comment'] = line.replace('Test Comment:', '').strip()
        elif line.startswith('Violates Rule:'):
            parsed_data['violates_rule'] = line.replace('Violates Rule:', '').strip()
    
    return parsed_data

async def process_single_rule(subreddit, rule_name, rule_description, kaggle_df):
    """Process a single rule - format it and generate realistic examples"""
    try:
        # Get random example comments from kaggle dataset
        example_comments = get_random_example_comments(kaggle_df)
        
        prompt = create_rule_processing_prompt(subreddit, rule_name, rule_description, example_comments)
        
        response = await model.generate_content_async(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9
            )
        )
        
        if response.candidates and response.candidates[0].content.parts:
            raw_response = response.candidates[0].content.parts[0].text
            parsed_data = parse_generated_response(raw_response)
            
            return {
                'Subreddit': subreddit,
                'Rule Name': rule_name,
                'Rule Description': rule_description,
                'Formatted Rule': parsed_data.get('formatted_rule'),
                'Positive Example 1': parsed_data.get('positive_example_1'),
                'Negative Example 1': parsed_data.get('negative_example_1'),
                'Positive Example 2': parsed_data.get('positive_example_2'),
                'Negative Example 2': parsed_data.get('negative_example_2'),
                'Test Comment': parsed_data.get('test_comment'),
                'Violates Rule': parsed_data.get('violates_rule'),
                'Raw Response': raw_response,
                'Example Comments Used': example_comments,
                'Error': None
            }
        else:
            return {
                'Subreddit': subreddit,
                'Rule Name': rule_name,
                'Rule Description': rule_description,
                'Formatted Rule': None,
                'Positive Example 1': None,
                'Negative Example 1': None,
                'Positive Example 2': None,
                'Negative Example 2': None,
                'Test Comment': None,
                'Violates Rule': None,
                'Raw Response': None,
                'Example Comments Used': example_comments,
                'Error': 'No response from model'
            }
            
    except Exception as e:
        return {
            'Subreddit': subreddit,
            'Rule Name': rule_name,
            'Rule Description': rule_description,
            'Formatted Rule': None,
            'Positive Example 1': None,
            'Negative Example 1': None,
            'Positive Example 2': None,
            'Negative Example 2': None,
            'Test Comment': None,
            'Violates Rule': None,
            'Raw Response': None,
            'Example Comments Used': None,
            'Error': str(e)
        }

async def process_batch_async(batch_data, kaggle_df):
    """Process a batch of rules with kaggle examples"""
    tasks = []
    for subreddit, rule_name, rule_description in batch_data:
        tasks.append(process_single_rule(subreddit, rule_name, rule_description, kaggle_df))
    
    print(f"Processing {len(tasks)} rules with kaggle examples...")
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
    """Main function to process rules and generate realistic examples"""
    # Load input data
    input_csv_path = './data/synthetic_generation/popular_subreddit_rules.csv'
    kaggle_csv_path = './data/synthetic_generation/kaggle_train_test.csv'  # Update this path
    
    print("Loading subreddit rules data...")
    rules_df = load_data(input_csv_path)
    rules_df = rules_df.sample(frac=1).reset_index(drop=True)
    rules_df = rules_df.iloc[:1000]  # Start with smaller sample

    print("Loading kaggle examples dataset...")
    kaggle_df = load_kaggle_examples(kaggle_csv_path)

    print(f"Processing {len(rules_df)} rules from {rules_df['Subreddit'].nunique()} subreddits")
    print(f"Using {len(kaggle_df)} example comments from kaggle dataset")
    
    # Show sample of input data
    print(f"\nSample input data:")
    print(rules_df.head(3)[['Subreddit', 'Rule Name', 'Rule Description']])
    
    print(f"\nSample kaggle examples:")
    print(kaggle_df.head(2))
    
    # Prepare data for processing
    rule_combinations = []
    for _, row in rules_df.iterrows():
        rule_combinations.append((row['Subreddit'], row['Rule Name'], row['Rule Description']))
    
    print(f"\nGenerating realistic Reddit-style examples using kaggle examples as inspiration...")
    
    all_results = []
    batch_size = 20  # Smaller batch for testing
    
    # Process in batches
    for i in range(0, len(rule_combinations), batch_size):
        batch = rule_combinations[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"\n--- Processing batch {batch_num} ({len(batch)} rules) ---")
        
        results = await process_batch_async(batch, kaggle_df)
        all_results.extend(results)

        # Save batch results
        if results:
            batch_df = pd.DataFrame(results)
            filename = f"kaggle_inspired_rules_batch_{batch_num}.csv"
            batch_df.to_csv(filename, index=False)
            print(f"Saved batch to {filename}")

    # Save all results
    if all_results:
        final_df = pd.DataFrame(all_results)
        output_path = 'kaggle_inspired_subreddit_rules_with_examples.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved all processed rules to {output_path}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total rules processed: {len(final_df)}")
        print(f"Successfully formatted: {final_df['Formatted Rule'].notna().sum()}")
        print(f"Examples generated: {final_df['Positive Example 1'].notna().sum()}")
        print(f"Success rate: {(final_df['Formatted Rule'].notna().sum() / len(final_df)) * 100:.1f}%")
        print(f"Unique subreddits: {final_df['Subreddit'].nunique()}")
        
        return final_df
    else:
        print("No results generated")
        return None

if __name__ == "__main__":
    result_df = asyncio.run(main())