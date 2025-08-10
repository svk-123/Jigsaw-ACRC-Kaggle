import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

def get_summary(df):
    """
    Evaluate classification performance for rule violation predictions.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'violates_rule' and 'predicted_rule_violation' columns
    
    Returns:
    dict: Summary of classification metrics
    """
    
    def convert_to_bool(series):
        """Simple boolean conversion"""
        if series.dtype == 'bool':
            return series
        
        def to_bool(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).strip().lower()
            if val_str in ['true', '1', 'yes', 'y']:
                return True
            elif val_str in ['false', '0', 'no', 'n']:
                return False
            else:
                return np.nan
        
        return series.apply(to_bool)
    
    # Convert both columns to boolean
    df_copy = df.copy()
    df_copy['violates_rule_bool'] = convert_to_bool(df_copy['violates_rule'])
    df_copy['predicted_bool'] = convert_to_bool(df_copy['predicted_rule_violation'])
    
    # Filter valid data (no NaNs, no errors)
    valid_mask = (
        df_copy['violates_rule_bool'].notna() & 
        df_copy['predicted_bool'].notna() &
        (df_copy['predicted_rule_violation'] != 'Error')
    )
    df_clean = df_copy[valid_mask]
    
    # Calculate basic statistics
    total_rows = len(df)
    valid_rows = len(df_clean)
    success_rate = (valid_rows / total_rows * 100) if total_rows > 0 else 0
    
    summary = {
        'total_rows': total_rows,
        'valid_rows': valid_rows,
        'success_rate': success_rate,
        'f1_score': 0,
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'classification_report': None
    }
    
    if valid_rows > 0:
        # Convert to numpy boolean arrays for sklearn
        y_true = np.array(df_clean['violates_rule_bool'], dtype=bool)
        y_pred = np.array(df_clean['predicted_bool'], dtype=bool)
        
        # Calculate metrics
        summary['f1_score'] = f1_score(y_true, y_pred)
        summary['accuracy'] = accuracy_score(y_true, y_pred)
        summary['precision'] = precision_score(y_true, y_pred, zero_division=0)
        summary['recall'] = recall_score(y_true, y_pred, zero_division=0)
        summary['classification_report'] = classification_report(
            y_true, y_pred, target_names=['No Violation', 'Violation'], output_dict=True
        )
    
    return summary

def print_summary(summary):
    """Pretty print the classification summary"""
    print(f"Total rows: {summary['total_rows']}")
    print(f"Valid rows: {summary['valid_rows']}")
    print(f"Success rate: {summary['success_rate']:.2f}%\n")
    
    if summary['valid_rows'] > 0:
        print(f"F1 Score: {summary['f1_score']:.4f}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        print(f"Precision: {summary['precision']:.4f}")
        print(f"Recall: {summary['recall']:.4f}\n")
        
        if summary['classification_report']:
            print("Classification Report:")
            from sklearn.metrics import classification_report
            # Reconstruct the report for pretty printing
            print("              precision    recall  f1-score   support")
            print()
            for label in ['No Violation', 'Violation']:
                metrics = summary['classification_report'][label]
                print(f"{label:>13} {metrics['precision']:10.2f} {metrics['recall']:9.2f} "
                      f"{metrics['f1-score']:9.2f} {metrics['support']:9.0f}")
            print()
            print(f"    accuracy                      {summary['classification_report']['accuracy']:.2f} "
                  f"{summary['classification_report']['macro avg']['support']:.0f}")
            print(f"   macro avg {summary['classification_report']['macro avg']['precision']:10.2f} "
                  f"{summary['classification_report']['macro avg']['recall']:9.2f} "
                  f"{summary['classification_report']['macro avg']['f1-score']:9.2f} "
                  f"{summary['classification_report']['macro avg']['support']:9.0f}")
            print(f"weighted avg {summary['classification_report']['weighted avg']['precision']:10.2f} "
                  f"{summary['classification_report']['weighted avg']['recall']:9.2f} "
                  f"{summary['classification_report']['weighted avg']['f1-score']:9.2f} "
                  f"{summary['classification_report']['weighted avg']['support']:9.0f}")
    else:
        print("No valid data for evaluation.")

# Usage:
# summary = get_classification_summary(df_test)
# print_summary(summary)