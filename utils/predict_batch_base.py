from tqdm import tqdm
import numpy as np

def predict_batch_base(model, df, batch_size=12):
    """Process dataframe using batch predictions with progress bar and error handling"""
    
    # Calculate number of batches
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    
    all_results = []
    failed_batches = []
    failed_indices = []
    
    # Process in batches with progress bar
    with tqdm(total=len(df), desc="Processing predictions") as pbar:
        for i in range(0, len(df), batch_size):
            # Get current batch
            batch_df = df.iloc[i:i+batch_size]
            current_batch_num = i//batch_size + 1
            
            try:
                # Convert batch to list of Series (input format for predict_batch)
                batch_list = [row for idx, row in batch_df.iterrows()]  # Fixed syntax
                
                # Get predictions for this batch
                batch_results = model.predict_batch(batch_list)
                
                # Add batch indices to results for tracking
                for j, result in enumerate(batch_results):
                    result['original_index'] = batch_df.index[j]
                
                # Add to results
                all_results.extend(batch_results)
                
            except Exception as e:
                # Log the error and continue with next batch
                tqdm.write(f"Error processing batch {current_batch_num}: {str(e)}")
                failed_batches.append(current_batch_num)
                batch_indices = batch_df.index.tolist()
                failed_indices.extend(batch_indices)
                
                # Create error results for all items in failed batch
                for idx in batch_indices:
                    error_result = {
                        'prediction': 'Error',
                        'is_violation': False,  # Default to False for errors
                        'violation_probability': 0.0,
                        'confidence': 0.0,
                        'original_index': idx,
                        'error': f"Batch {current_batch_num} failed: {str(e)}",
                        'batch_error': True
                    }
                    all_results.append(error_result)
            
            # Update progress bar
            pbar.update(len(batch_df))
            pbar.set_postfix({
                'Batch': f'{current_batch_num}/{num_batches}',
                'Failed': len(failed_batches)
            })
    
    # Print summary
    if failed_batches:
        print(f"\nProcessing completed with {len(failed_batches)} failed batches.")
        print(f"Failed batch numbers: {failed_batches}")
        print(f"Total failed rows: {len(failed_indices)}")
    else:
        print(f"\nAll {num_batches} batches processed successfully!")
    
    return all_results, failed_batches, failed_indices