import os
import pandas as pd
import zipfile
import shutil
from datasets import Dataset, Features, Image, Value
import json


def extract_unique_ids(input_file, output_file):
    """
    Extract unique IDs from an input CSV file and save them to an output CSV file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file where unique IDs will be saved.
    """
    df = pd.read_csv(input_file, usecols=['unique_id'])  

    df.to_csv(output_file, index=False)

    print(f"Unique IDs extracted from {input_file} and saved to {output_file}")


def remove_missing(file_path, model):
    """
    Removes rows where f'{model}' is missing or contains values other than 'Likely' or 'Unlikely'.
    Also removes those same 'unique_id' rows from the secondary file.
    Saves both cleaned CSV files back to their original paths.
    
    Parameters:
    - file_path: Path to the main CSV file (which includes the f'{model}' column).
    - secondary_file_path: Path to the second CSV file containing only 'unique_id'.
    """
    df = pd.read_csv(file_path)
    print(f"Original shape of main file: {df.shape}")

    valid_values = ["Likely", "Unlikely"]

    invalid_rows = df[(df[f'{model}'].isna()) | (~df[f'{model}'].isin(valid_values))]

    if not invalid_rows.empty:
        print("Rows with missing or invalid values in f'{model}':")
        print(invalid_rows['unique_id'].tolist())

        invalid_unique_ids = invalid_rows['unique_id'].tolist()

        df_cleaned = df[~df['unique_id'].isin(invalid_unique_ids)]
        print(f"Cleaned shape of main file: {df_cleaned.shape}")

        df_cleaned.to_csv(file_path, index=False)
        print(f"Saved cleaned main file to {file_path}")
    else:
        print("No invalid rows found.")
        df.to_csv(file_path, index=False)
        print(f"Saved unchanged main file to {file_path}")

    return df_cleaned if not invalid_rows.empty else df


def remove_duplicate_fields(file_path, column_name):
    """
    Removes rows with duplicate 'unique_id' values.
    Prints the duplicate ids and their count, then saves the cleaned CSV back to the original file.
    """
    df = pd.read_csv(file_path)

    duplicate_rows = df[df.duplicated(subset=column_name, keep=False)]

    if not duplicate_rows.empty:
        print("Duplicate unique_ids and their counts:")
        print(duplicate_rows[column_name].value_counts())
    else:
        print("No duplicates in the csv")

    df_cleaned = df.drop_duplicates(subset=column_name, keep='first')

    df_cleaned.to_csv(file_path, index=False)

    return df_cleaned


def remove_duplicate_images(folderpath):
    """Remove duplicate images based on their file names (ignoring case and extensions)."""
    seen_files = {}

    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)

        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            base_name = os.path.splitext(filename)[0].lower()

            if base_name in seen_files:
                print(f"Duplicate found: {filename} -> Removing")
                os.remove(filepath)
            else:
                seen_files[base_name] = filepath

    print("Duplicate image removal complete.")


def merge_csv_files(csv_files, final_columns, save_directory, output_filename='text_merged.csv'):
    """
    Merges and cleans multiple CSV files, removes duplicates, and saves the final CSV.

    Args:
        csv_files (list): List of CSV file paths to merge.
        final_columns (list): List of columns to retain in the final CSV.
        save_directory (str): Directory where the final CSV will be saved.
        output_filename (str): Name of the output CSV file (default is 'text_merged.csv').
    """
    def load_and_clean_csv(file_path):
        """Loads and cleans a single CSV file."""
        try:
            df = pd.read_csv(file_path)

            if 'content' in df.columns:
                df['content'] = df['content'].replace('\n', ' ', regex=True)

            if 'first_paragraph' in df.columns:
                df = df.drop(columns=['first_paragraph'])

            df = df[final_columns]
            return df
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return pd.DataFrame(columns=final_columns)

    merged_data = pd.DataFrame(columns=final_columns)

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = load_and_clean_csv(csv_file)
            merged_data = pd.concat([merged_data, df], ignore_index=True)
        else:
            print(f"File {csv_file} not found.")

    duplicate_ids = merged_data['unique_id'].value_counts()
    duplicates = duplicate_ids[duplicate_ids > 1]

    if not duplicates.empty:
        print("Duplicate unique_ids found:")
        for unique_id, count in duplicates.items():
            print(f"unique_id: {unique_id}, count: {count}")
        
        merged_data = merged_data.drop_duplicates(subset=['unique_id'], keep='first')
    else:
        print("No duplicate unique_ids found.")

    output_file = os.path.join(save_directory, output_filename)
    merged_data.to_csv(output_file, index=False)
    print(f"Merging complete. Final CSV saved as '{output_file}'.")

def finalize_datasets():
    csv_files = [
        '/fs01/projects/NMB-Plus/Caesar/Datasets/merged_output.csv',
        '/fs01/projects/NMB-Plus/Caesar/Datasets/merged_output_images.csv'
    ]

    merged_df = pd.read_csv(csv_files[0], on_bad_lines='skip')

    for file in csv_files[1:]:
        df = pd.read_csv(file, on_bad_lines='skip')
        
        common_columns = merged_df.columns.intersection(df.columns).tolist()
        common_columns.remove('unique_id') 
        
        df = df.drop(columns=common_columns)
        
        merged_df = pd.merge(merged_df, df, on='unique_id', how='inner')

    merged_df.to_csv('final.csv', index=False)

    print("Merged CSV saved to 'final.csv'")


def text_majority_vote(input_file, output_file):
    df = pd.read_csv(input_file)

    vote_columns = ['gemma', 'gpt4omini', 'llama3.1', 'llama3', 'mistral']

    majority_votes = []

    for i, row in df.iterrows():
        votes = [row[col].strip() for col in vote_columns if col in row and pd.notna(row[col])]
        if not votes:  
            majority_votes.append(None)
            continue
        
        vote_count = {label: votes.count(label) for label in set(votes)}
        
        majority_label = max(vote_count, key=vote_count.get)
        
        majority_votes.append(majority_label)

    df['text_majority_vote'] = majority_votes

    df.to_csv(output_file, index=False)
    print(f"Majority voting completed and saved to '{output_file}'")
    

def image_majority_vote(input_file, output_file):
    df = pd.read_csv(input_file)

    vote_columns = ['phi_label', 'cpm_label']

    majority_votes = []

    for i, row in df.iterrows():
        votes = [row[col].strip() for col in vote_columns if col in row and pd.notna(row[col])]
        
        if not votes:  
            majority_votes.append(None)  
            continue
        
        vote_count = {label: votes.count(label) for label in set(votes)}
        
        max_count = max(vote_count.values())
        max_labels = [label for label, count in vote_count.items() if count == max_count]

        majority_label = max_labels[0] if len(max_labels) == 1 else 'phi_label' if 'phi_label' in max_labels else max_labels[0]

        majority_votes.append(majority_label)

    df['image_majority_vote'] = majority_votes

    df.to_csv(output_file, index=False)
    print(f"Majority voting completed and saved to '{output_file}'")


def final_label(input_file, output_file):
    df = pd.read_csv(input_file)

    final_labels = []

    for i, row in df.iterrows():
        text_vote = row['text_majority_vote']
        image_vote = row['image_majority_vote']
        
        if text_vote == 'Likely' or image_vote == 'Likely':
            final_labels.append('Likely')
        else:
            final_labels.append('Unlikely')

    df['label'] = final_labels

    df.to_csv(output_file, index=False)
    print(f"Final labels added and saved to '{output_file}'")

def remove_columns(input_file, output_file):
    try:
        df = pd.read_csv(input_file)

        columns_to_remove = [
            'image'
        ]

        missing_cols = [col for col in columns_to_remove if col not in df.columns]
        if missing_cols:
            print(f"Warning: The following columns were not found in the DataFrame: {missing_cols}")

        df_cleaned = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

        df_cleaned.to_csv(output_file, index=False)
        print(f"Columns removed and saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The input file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def clean_csv_and_images(input_csv, output_csv, images_dir):
    df = pd.read_csv(input_csv)

    unique_ids = set(df['unique_id'].astype(str))

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg']

    rows_to_remove = []
    image_count = {}
    
    for index, row in df.iterrows():
        unique_id = str(row['unique_id'])
        image_found = False
        
        for ext in image_extensions:
            image_path = os.path.join(images_dir, f"{unique_id}{ext}")
            if os.path.exists(image_path):
                image_found = True
                image_count[unique_id] = image_count.get(unique_id, 0) + 1  
                break  

        if not image_found:
            rows_to_remove.append(index)
            print(f"Removed unique_id: {unique_id}")

    df_cleaned = df.drop(index=rows_to_remove)
    df_cleaned.to_csv(output_csv, index=False)

    for unique_id, count in image_count.items():
        if count > 1:
            print(f"Warning: Multiple images found for unique_id {unique_id}: {count} images.")
            image_count[unique_id] = 1  

    existing_images = set()
    for img in os.listdir(images_dir):
        name, ext = os.path.splitext(img)
        if ext.lower() in image_extensions:
            existing_images.add(name)  

    print("Existing images in directory:", existing_images)

    images_to_remove = existing_images - unique_ids
    
    for image_name in images_to_remove:
        print(f"Images to remove for unique_id not in CSV: {image_name}")
        for ext in image_extensions:
            image_path = os.path.join(images_dir, f"{image_name}{ext}")
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed image: {image_path}")
                break  

    for unique_id, count in image_count.items():
        if count > 1:
            print(f"Removing extra images for unique_id: {unique_id}")
            for ext in image_extensions:
                extra_image_path = os.path.join(images_dir, f"{unique_id}{ext}")
                if os.path.exists(extra_image_path):
                    os.remove(extra_image_path)
                    print(f"Removed extra image: {extra_image_path}")

    print(f"Cleaned CSV saved to '{output_csv}' and unused images removed from '{images_dir}'")


def csv_to_parquet(input_csv, output_parquet):
    df = pd.read_csv(input_csv)
    df.to_parquet(output_parquet, index=False)
    print(f"Converted '{input_csv}' to '{output_parquet}' successfully.")


def csv_to_json(input_csv, output_json):
    df = pd.read_csv(input_csv)
    df.to_json(output_json, orient='records', lines=True)
    print(f"Converted '{input_csv}' to '{output_json}' successfully.")


def zip_files(input_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.isdir(input_path):
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(input_path)))
        elif os.path.isfile(input_path):
            zipf.write(input_path, os.path.basename(input_path))
        else:
            print(f"The path '{input_path}' is not a valid file or directory.")
            return

    print(f"Created zip file '{output_zip}' successfully.")


def sample_csv_and_copy_images(input_csv, output_csv, images_dir, sample_size=1000):
    df = pd.read_csv(input_csv)

    if sample_size > len(df):
        print(f"Sample size {sample_size} is greater than the number of rows in the CSV ({len(df)}).")
        sample_size = len(df) 

    df_sampled = df.sample(n=sample_size, random_state=1)  

    df_sampled.to_csv(output_csv, index=False)
    print(f"Random sample of {sample_size} rows saved to '{output_csv}'.")

    sample_images_dir = 'sample_images'
    os.makedirs(sample_images_dir, exist_ok=True)

    for unique_id in df_sampled['unique_id']:
        for ext in ['.jpg', '.jpeg', '.png','.gif', '.bmp', '.tiff', '.webp', '.svg']:  
            image_path = os.path.join(images_dir, f"{unique_id}{ext}")
            if os.path.exists(image_path):
                shutil.copy(image_path, sample_images_dir)
                print(f"Copied image for unique_id {unique_id} to '{sample_images_dir}'.")
                break 

    print(f"Copied images for the sampled rows into '{sample_images_dir}'.")

def convert_json_structure(input_json, output_json):
    json_objects = []

    with open(input_json, 'r') as file:
        for line in file:
            if line.strip():  
                json_objects.append(json.loads(line))

    with open(output_json, 'w') as file:
        json.dump(json_objects, file, indent=4)

    print(f"Converted JSON saved to '{output_json}'.")


def add_file_name_to_csv(input_csv, output_csv, images_dir, extensions=['.jpg', '.jpeg', '.webp', '.png', '.gif']):
    df = pd.read_csv(input_csv)

    for index, row in df.iterrows():
        unique_id = row.get('unique_id')

        if unique_id:
            found_image = False
            
            for ext in extensions:
                image_name = f"{unique_id}{ext}"
                image_path = os.path.join(images_dir, image_name)
                
                if os.path.exists(image_path):
                    relative_path = os.path.relpath(image_path, start=os.path.dirname(output_csv))
                    df.at[index, 'image'] = relative_path  
                    found_image = True
                    break  

            if not found_image:
                df.at[index, 'image'] = None  

    df.to_csv(output_csv, index=False)

    print(f"Modified CSV saved to '{output_csv}'.")




def reorder_csv_columns(input_csv, output_csv, column_order):
    df = pd.read_csv(input_csv)

    df = df[column_order]

    df.to_csv(output_csv, index=False)

    print(f"Reordered CSV saved to '{output_csv}'.")



def json_to_csv(input_json, output_csv):
    with open(input_json, 'r') as file:
        json_data = json.load(file)

    df = pd.DataFrame(json_data)

    df.to_csv(output_csv, index=False)

    print(f"Converted JSON saved to '{output_csv}'.")


def create_preview():
    df = pd.read_csv('sample_dataset.csv')
    features = Features({
    'unique_id': Value('string'),
    'outlet': Value('string'),
    'title': Value('string'),
    'date_published': Value('string'),
    'canonical_link': Value('string'),
    'source_url': Value('string'),
    'content': Value('string'),
    'image': Image(),
    'image_description': Value('string'),
    'text_label': Value('string'),
    'multimodal_label': Value('string')
    })

    dataset = Dataset.from_pandas(df, features=features)
    print("Hugging Face Dataset created successfully.")

    dataset.push_to_hub("POLLCHECK/Dataset", private=False)


if __name__ == "__main__":
    zip_files("images", "images.zip")