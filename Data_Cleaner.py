import pandas as pd
import numpy as np
from scipy import stats

def load_csv(file_path):

    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        return df
    except FileNotFoundError:
        print("File not found. Please check the file path and try again.")
        return None

def clean_missing_values(df, strategy='mean'):

    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            print(f"Handling {missing_count} missing values in column: {column}")
            
            if df[column].dtype == 'object':

                most_frequent = df[column].mode()[0]
                df[column].fillna(most_frequent, inplace=True)
                print(f"Filled missing values in column '{column}' with mode: {most_frequent}")
            else:

                if strategy == 'mean':
                    mean_value = df[column].mean()
                    df[column].fillna(mean_value, inplace=True)
                    print(f"Filled missing values in column '{column}' with mean: {mean_value}")
                elif strategy == 'median':
                    median_value = df[column].median()
                    df[column].fillna(median_value, inplace=True)
                    print(f"Filled missing values in column '{column}' with median: {median_value}")
                elif strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
                    print(f"Dropped rows with missing values in column '{column}'")

    return df

def remove_outliers(df, z_thresh=3):

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    initial_count = df.shape[0]

    z_scores = np.abs(stats.zscore(df[numerical_cols]))
    df = df[(z_scores < z_thresh).all(axis=1)]
    removed_count = initial_count - df.shape[0]

    print(f"Removed {removed_count} outliers from the dataset.")
    return df

def clean_inconsistent_entries(df):

    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        original_unique_count = df[col].nunique()
        df[col] = df[col].str.strip().str.lower()
        cleaned_unique_count = df[col].nunique()
        if original_unique_count != cleaned_unique_count:
            print(f"Column '{col}' cleaned for inconsistent entries. Unique values reduced from {original_unique_count} to {cleaned_unique_count}.")

    return df

def save_cleaned_data(df, output_path):

    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

def main():

    input_file = r'User_product_purchase_details_p2.csv'
    output_file = r'output2_csv.csv'  

    df = load_csv(input_file)
    if df is None:
        return

    df = clean_missing_values(df, strategy='mean')
    df = remove_outliers(df, z_thresh=3)
    df = clean_inconsistent_entries(df)


    save_cleaned_data(df, output_file)

if __name__ == "__main__":
    main()
