import google.generativeai as genai
import asyncio
import os
import pandas as pd
import random

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Set random seed for reproducibility
random.seed(42)

def load_data():
    """Load subreddits and rules from CSV files"""
    try:
        # Load subreddits - assuming CSV has a 'subreddit' column
        subreddits_df = pd.read_csv('subreddits.csv')
        subreddits = subreddits_df['subreddit'].tolist()
        
        # Load rules - assuming CSV has a 'rule' column  
        rules_df = pd.read_csv('rules.csv')
        rules = rules_df['rule'].tolist()
        
        print(f"Loaded {len(subreddits)} subreddits and {len(rules)} rules")
        return subreddits, rules
        
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Make sure 'subreddits.csv' and 'rules.csv' exist")
        raise

def create_prompt(subreddit, rule):
    """Create the synthetic data generation prompt"""
    return f"""Generate synthetic Reddit comment moderation data for /r/{subreddit}.

Create 4 example comments and 1 test comment for this rule: "{rule}"

Format your response as:
Positive Example 1: [comment that violates the rule]
Negative Example 1: [comment that doesn't violate the rule]  
Positive Example 2: [comment that violates the rule]
Negative Example 2: [comment that doesn't violate the rule]
Test Comment: [comment for testing - randomly violates or doesn't violate the rule]

Make the comments realistic for /r/{subreddit} and clearly show rule violations vs non-violations."""

async def generate_synthetic_data(subreddit, rule):
    """Generate synthetic moderation data using Gemini API"""
    try:
        prompt = create_prompt(subreddit, rule)
        response = await model.generate_content_async(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.7)
        )
        
        if response.candidates and response.candidates[0].content.parts:
            return {
                'subreddit': subreddit,
                'rule': rule, 
                'generated_data': response.candidates[0].content.parts[0].text,
                'error': None
            }
        else:
            return {'subreddit': subreddit, 'rule': rule, 'generated_data': None, 'error': 'No response'}
            
    except Exception as e:
        return {'subreddit': subreddit, 'rule': rule, 'generated_data': None, 'error': str(e)}

async def process_batch_async(batch_data):
    """Process a batch of subreddit/rule combinations"""
    tasks = []
    for subreddit, rule in batch_data:
        tasks.append(generate_synthetic_data(subreddit, rule))
    
    print(f"Processing {len(tasks)} synthetic data generations...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("Batch completed.")
    
    return results

async def main():
    """Main function to generate synthetic data"""
    # Load data from CSV files
    subreddits, rules = load_data()
    
    # Generate random combinations
    num_samples = 200  # Adjust as needed
    batch_size = 10
    
    combinations = []
    for _ in range(num_samples):
        subreddit = random.choice(subreddits)
        rule = random.choice(rules)
        combinations.append((subreddit, rule))
    
    print(f"Generating {num_samples} synthetic samples with seed=42...")
    print(f"Sample combinations: {combinations[:3]}...")  # Show first 3
    
    all_results = []
    
    # Process in batches
    for i in range(0, len(combinations), batch_size):
        batch = combinations[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"\n--- Processing batch {batch_num} ---")
        
        results = await process_batch_async(batch)
        all_results.extend(results)
        
        # Save batch results
        batch_df = pd.DataFrame(results)
        filename = f"synthetic_batch_{batch_num}.csv"
        batch_df.to_csv(filename, index=False)
        print(f"Saved batch to {filename}")
    
    # Save all results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv('synthetic_moderation_data.csv', index=False)
    print(f"\nGenerated {len(final_df)} synthetic samples total")
    
    # Print summary stats
    print(f"Unique subreddits used: {final_df['subreddit'].nunique()}")
    print(f"Unique rules used: {final_df['rule'].nunique()}")
    print(f"Success rate: {(final_df['error'].isna().sum() / len(final_df)) * 100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())