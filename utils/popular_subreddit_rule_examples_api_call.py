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
#model = genai.GenerativeModel("gemini-2.0-flash")
model = genai.GenerativeModel("gemini-2.5-flash")

# Set random seed for reproducibility
random.seed(149831)

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

def create_rule_processing_prompt(subreddit, rule_name, rule_description):
    """Create prompt to format rule and generate realistic Reddit examples"""
    return f"""You are processing Reddit moderation rules for /r/{subreddit}.

Given this rule:
Rule Name: {rule_name}
Rule Description: {rule_description}

Perform these tasks:

1. Create a concise, clear formatted version of this rule (around 8-12 words) that captures the essence for moderation purposes.

2. Generate REALISTIC Reddit comment examples for /r/{subreddit}. Make about 50% of them use authentic Reddit language including:
   - Internet slang (lit, fam, sus, cringe, based, etc.)
   - Poor grammar/spelling
   - Emojis 🔥💯😂
   - Self-promotion attempts
   - Spam-like content with suspicious links
   - Gaming references and memes
   - Casual/crude language
   - ALL CAPS text
   - Random promotional content
   - Adult content hints (but keep it PG-13)
   - Drug/pharmaceutical spam style
   - Clickbait language

Examples of realistic Reddit style:
"NEW RAP GROUP CHECK US OUT https://soundcloud.com/user-125895482"
"Get up to 15% Discount on pain killers Without Prescription"
"This mixtape is lit FAM! 🔥🔥💥"
"watch hooters best therein http://clickand.co/5agw"
"Free paypal cards here!! https://www.pointsprizes.com/ref/13226"
"Just made my account last night follow up on my photos 😘"

Format your response EXACTLY like this:
Formatted Rule: [concise 1-line rule in 10-20 words]
Positive Example 1: [realistic Reddit comment that violates the rule]
Negative Example 1: [realistic Reddit comment that doesn't violate the rule]
Positive Example 2: [realistic Reddit comment that violates the rule]
Negative Example 2: [realistic Reddit comment that doesn't violate the rule]
Test Comment: [realistic Reddit comment for testing - randomly violates or doesn't violate the rule]
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

async def process_single_rule(subreddit, rule_name, rule_description):
    """Process a single rule - format it and generate realistic examples"""
    try:
        prompt = create_rule_processing_prompt(subreddit, rule_name, rule_description)
        
        response = await model.generate_content_async(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,  # Higher temperature for more creative/chaotic examples
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
    """Main function to process rules and generate realistic examples"""
    # Load input data
    input_csv_path = './data/synthetic_generation/popular_subreddit_rules.csv'  # Update this path to your CSV file
    
    print("Loading subreddit rules data...")
    rules_df = load_data(input_csv_path)
    rules_df = rules_df.sample(frac=1).reset_index(drop=True)
    rules_df=rules_df.iloc[:1500]

    print(f"Processing {len(rules_df)} rules from {rules_df['Subreddit'].nunique()} subreddits")
    
    # Show sample of input data
    print(f"\nSample input data:")
    print(rules_df.head(3)[['Subreddit', 'Rule Name', 'Rule Description']])
    
    # Prepare data for processing
    rule_combinations = []
    for _, row in rules_df.iterrows():
        rule_combinations.append((row['Subreddit'], row['Rule Name'], row['Rule Description']))
    
    print(f"\nGenerating realistic Reddit-style examples for {len(rule_combinations)} rules...")
    print("Examples will include slang, spam, self-promotion, and authentic Reddit chaos! 🔥")
    
    all_results = []
    batch_size = 30  # Smaller batch size since we're generating more complex content
    
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
            filename = f"realistic_rules_batch_{batch_num}.csv"
            batch_df.to_csv(filename, index=False)
            print(f"Saved batch to {filename}")

    # Save all results
    if all_results:
        final_df = pd.DataFrame(all_results)
        output_path = 'realistic_subreddit_rules_with_examples.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved all processed rules to {output_path}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total rules processed: {len(final_df)}")
        print(f"Successfully formatted: {final_df['Formatted Rule'].notna().sum()}")
        print(f"Examples generated: {final_df['Positive Example 1'].notna().sum()}")
        print(f"Success rate: {(final_df['Formatted Rule'].notna().sum() / len(final_df)) * 100:.1f}%")
        print(f"Unique subreddits: {final_df['Subreddit'].nunique()}")
        
        # Show sample results with realistic examples
        print(f"\nSample realistic Reddit examples:")
        sample_df = final_df[final_df['Formatted Rule'].notna()].head(3)
        for _, row in sample_df.iterrows():
            print(f"\n/r/{row['Subreddit']}:")
            print(f"  Formatted Rule: {row['Formatted Rule']}")
            print(f"  Violating Example: {row['Positive Example 1']}")
            print(f"  Good Example: {row['Negative Example 1']}")
        
        print(f"\nOutput columns: {list(final_df.columns)}")
        print(f"Output shape: {final_df.shape}")
        
        return final_df
    else:
        print("No results generated")
        return None

if __name__ == "__main__":
    result_df = asyncio.run(main())