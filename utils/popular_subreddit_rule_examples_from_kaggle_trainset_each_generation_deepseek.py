from openai import AsyncOpenAI
import asyncio
import os
import pandas as pd
import random

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

# Set random seed for reproducibility
random.seed(1436131)

def load_data(csv_path):
    """Load subreddit rules data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['formatted_rule']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"Loaded {len(df)} rules")
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
                           'negative_example_1', 'negative_example_2','test_comment','violates_rule']
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
Rule Context: {random_row['rule']}

Examples that Violate Rules:
- "{random_row['positive_example_1']}"
- "{random_row['positive_example_2']}"

Examples that does not Violate Rules:
- "{random_row['negative_example_1']}"
- "{random_row['negative_example_2']}"

test_comment: {random_row['test_comment']}
Violates Rule: {random_row['violates_rule']}
"""
    
    return example_comments

def create_rule_processing_prompt(subreddit, formatted_rule, example_comments, force_violation):
    """Create prompt to format rule and generate realistic Reddit examples"""

    # Determine test comment instruction based on force_violation parameter
    if force_violation is True:
        test_instruction = "Test Comment: [NEW realistic Reddit comment that VIOLATES the rule]"
        violates_instruction = 'Violates Rule: "Yes"'
    elif force_violation is False:
        #print("force violation is False")
        test_instruction = "Test Comment: [NEW realistic Reddit comment that does NOT violate the rule]"
        violates_instruction = 'Violates Rule: "No"'
    else:
        #print("force violation is None")
        test_instruction = "Test Comment: [NEW realistic Reddit comment for testing - randomly violates or doesn't violate the rule]"
        violates_instruction = 'Violates Rule: "Yes" or "No" for the test comment'

    return f"""You are processing Reddit moderation rules for /r/{subreddit}.

Given this rule:
Rule: {formatted_rule}

Use these REAL Reddit comment examples as inspiration for style and authenticity:
{example_comments}

Based on these real examples, perform these tasks:

Generate NEW realistic Reddit comment examples for /r/{subreddit} and Rule: {formatted_rule} that follow similar patterns to the provided examples. Make them authentic with Reddit language.

Learn from the style and patterns in the provided real examples to create new, similar examples. Keep commnets with 10-50 words length.

Format your response EXACTLY like this:
Formatted Rule: [original rule]
Example 1: [NEW realistic Reddit comment that violates the rule, inspired by the real examples]
Example 2: [NEW realistic Reddit comment that violates the rule, inspired by the real examples]
Example 3: [NEW realistic Reddit comment that doesn't violate the rule]
Example 4: [NEW realistic Reddit comment that doesn't violate the rule]
{test_instruction}
{violates_instruction}

Make the comments feel authentic to /r/{subreddit} community with real Reddit chaos, slang, and behavior patterns with words from 20-60 in each example and test comment."""

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
        elif line.startswith('Example 1:'):
            parsed_data['positive_example_1'] = line.replace('Example 1:', '').strip()
        elif line.startswith('Example 2:'):
            parsed_data['positive_example_2'] = line.replace('Example 2:', '').strip()
        elif line.startswith('Example 3:'):
            parsed_data['negative_example_1'] = line.replace('Example 3:', '').strip()
        elif line.startswith('Example 4:'):
            parsed_data['negative_example_2'] = line.replace('Example 4:', '').strip()
        elif line.startswith('Test Comment:'):
            parsed_data['test_comment'] = line.replace('Test Comment:', '').strip()
        elif line.startswith('Violates Rule:'):
            parsed_data['violates_rule'] = line.replace('Violates Rule:', '').strip()
    
    return parsed_data

async def process_single_rule(subreddit, formatted_rule, kaggle_df, force_violation=False):
    """Process a single rule - format it and generate realistic examples"""
    try:
        # Get random example comments from kaggle dataset
        example_comments = get_random_example_comments(kaggle_df)

        prompt = create_rule_processing_prompt(subreddit, formatted_rule, example_comments, force_violation)

        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that processes Reddit moderation rules and generates realistic examples."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            stream=False
        )

        if response.choices and response.choices[0].message.content:
            raw_response = response.choices[0].message.content
            parsed_data = parse_generated_response(raw_response)
            
            return {
                'subreddit': subreddit,
                'rule': formatted_rule,
                'formatted_rule': parsed_data.get('formatted_rule'),
                'positive_example_1': parsed_data.get('positive_example_1'),
                'negative_example_1': parsed_data.get('negative_example_1'),
                'positive_example_2': parsed_data.get('positive_example_2'),
                'negative_example_2': parsed_data.get('negative_example_2'),
                'test_comment': parsed_data.get('test_comment'),
                'violates_rule': parsed_data.get('violates_rule'),
                'raw_response': raw_response,
                'example_comments_used': example_comments,
                'error': None
            }
        else:
            return {
                'subreddit': subreddit,
                'rule': formatted_rule,
                'formatted_rule': None,
                'positive_example_1': None,
                'negative_example_1': None,
                'positive_example_2': None,
                'negative_example_2': None,
                'test_comment': None,
                'violates_rule': None,
                'raw_response': None,
                'example_comments_used': example_comments,
                'error': 'No response from model'
            }
            
    except Exception as e:
        return {
            'subreddit': subreddit,
            'rule': formatted_rule,
            'formatted_rule': None,
            'positive_example_1': None,
            'negative_example_1': None,
            'positive_example_2': None,
            'negative_example_2': None,
            'test_comment': None,
            'violates_rule': None,
            'raw_response': None,
            'example_comments_used': None,
            'error': str(e)
        }

async def process_batch_async(batch_data, kaggle_df):
    """Process a batch of rules with kaggle examples"""
    tasks = []
    for subreddit, formatted_rule in batch_data:
        tasks.append(process_single_rule(subreddit, formatted_rule, kaggle_df))
    
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

async def main(sample_size=None):
    """Main function to process rules and generate realistic examples

    Args:
        sample_size: Number of rules to sample (1000-10000). If None, uses all data.
    """
    # Load input data
    input_csv_rules= '/home/vino/ML_Projects/Jigsaw-ACRC-Kaggle/data/synthetic_generation/community_rules.csv'
    input_csv_subreddits= '/home/vino/ML_Projects/Jigsaw-ACRC-Kaggle/data/synthetic_generation/subreddit_kaggle_list.csv'
    input_examples_csv_path = '/home/vino/ML_Projects/Jigsaw-ACRC-Kaggle/data/synthetic_generation/batch_0_train.csv'

    print("Loading subreddit rules data...")
    rules_df = load_data(input_csv_rules)

    print("Loading subreddit list...")
    subreddits_df = pd.read_csv(input_csv_subreddits)

    print("Loading kaggle examples dataset...")
    kaggle_examples_df = load_kaggle_examples(input_examples_csv_path)

    print(f"Loaded {len(rules_df)} total rules")
    print(f"Loaded {len(subreddits_df)} subreddits from subreddit list")
    print(f"Loaded {len(kaggle_examples_df)} total example comments from kaggle dataset")

    # Create combinations of all subreddits with all rules
    print("Creating subreddit-rule combinations...")
    all_combinations = []
    for _, subreddit_row in subreddits_df.iterrows():
        subreddit = subreddit_row['subreddit'] if 'subreddit' in subreddits_df.columns else subreddit_row.iloc[0]
        for _, rule_row in rules_df.iterrows():
            formatted_rule = rule_row['formatted_rule'] if 'formatted_rule' in rules_df.columns else rule_row.iloc[0]
            all_combinations.append((subreddit, formatted_rule))

    print(f"Created {len(all_combinations)} total subreddit-rule combinations")

    # Generic sampling logic - works with any dataset size
    if sample_size is not None:
        # Ensure sample_size doesn't exceed available data
        actual_sample_size = min(sample_size, len(all_combinations))

        if actual_sample_size < len(all_combinations):
            print(f"Sampling {actual_sample_size} combinations from {len(all_combinations)} total combinations...")
            all_combinations = random.sample(all_combinations, actual_sample_size)
            print(f"Selected {len(all_combinations)} subreddit-rule combinations")
        else:
            print(f"Using all {len(all_combinations)} available combinations (requested: {sample_size})")

    # Always use all available data for maximum diversity
    print(f"Using all {len(kaggle_examples_df)} kaggle examples for inspiration")

    print(f"Processing {len(all_combinations)} subreddit-rule combinations")
    print(f"Using {len(kaggle_examples_df)} example comments from kaggle dataset")
    
    # Show sample of input data
    print(f"\nSample input data:")
    print(f"First 3 combinations: {all_combinations[:3]}")

    print(f"\nSample kaggle examples:")
    print(kaggle_examples_df.head(2))

    print(f"\nGenerating realistic Reddit-style examples using kaggle examples as inspiration...")

    all_results = []
    batch_size = 200  # Smaller batch for testing

    # Process in batches
    for i in range(0, len(all_combinations), batch_size):
        batch = all_combinations[i:i + batch_size]
        batch_num = i // batch_size 
        
        print(f"\n--- Processing batch {batch_num} ({len(batch)} rules) ---")
        
        results = await process_batch_async(batch, kaggle_examples_df)
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
        print(f"\nSaved all processed rules to {output_path}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total rules processed: {len(final_df)}")
        print(f"Successfully formatted: {final_df['formatted_rule'].notna().sum()}")
        print(f"Examples generated: {final_df['positive_example_1'].notna().sum()}")
        print(f"Success rate: {(final_df['formatted_rule'].notna().sum() / len(final_df)) * 100:.1f}%")
        print(f"Unique subreddits: {final_df['subreddit'].nunique()}")
        
        return final_df
    else:
        print("No results generated")
        return None

if __name__ == "__main__":
    # Example usage:
    # For 2000 samples: result_df = asyncio.run(main(2000))
    # For 5000 samples: result_df = asyncio.run(main(5000))
    # For all data: result_df = asyncio.run(main())
    result_df = asyncio.run(main(5000))