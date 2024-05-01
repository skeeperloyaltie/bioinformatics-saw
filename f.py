import pandas as pd
import os
import subprocess
import tempfile
import logging

# Configure logging to file and console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s',
                    handlers=[
                        logging.FileHandler("data_processing.log"),
                        logging.StreamHandler()
                    ])

def load_data(file_path, sequence_column):
    """Loads data from CSV and drops rows with missing sequence data."""
    try:
        df = pd.read_csv(file_path)
        initial_count = len(df)
        df = df.dropna(subset=[sequence_column])
        logging.info(f"Loaded data from {file_path}. Dropped {initial_count - len(df)} rows due to missing values.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise

def predict_structure_with_ipknot(sequence, sequence_id):
    """Predicts RNA secondary structure using IPknot."""
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(f">{sequence_id}\n{sequence}\n")
            temp_file.seek(0)
            result = subprocess.run(['ipknot', temp_file_name], capture_output=True, text=True, check=True)
            return result.stdout.split('\n')[2]
    except subprocess.CalledProcessError as e:
        logging.warning(f"Error in predicting structure for sequence ID {sequence_id}: {e}")
        return None
    finally:
        os.unlink(temp_file_name)

def process_data(df, sequence_column, id_column, output_path):
    """Processes structures, tracks progress, and handles duplicates and interruptions."""
    if os.path.exists(output_path):
        processed_df = pd.read_csv(output_path)
    else:
        processed_df = pd.DataFrame(columns=list(df.columns) + [f"{sequence_column}_structure"])

    # Remove duplicates based on the sequence column
    df = df.drop_duplicates(subset=[sequence_column])
    logging.info(f"Removed duplicates. Processing {len(df)} unique sequences.")

    for index, row in df.iterrows():
        if pd.isnull(row[sequence_column]):
            logging.warning(f"Skipping null sequence for ID {row[id_column]}")
            continue
        structure = predict_structure_with_ipknot(row[sequence_column], row[id_column])
        if structure:
            row[f"{sequence_column}_structure"] = structure
            processed_df = pd.concat([processed_df, pd.DataFrame([row])], ignore_index=True)
            processed_df.to_csv(output_path, index=False)
            logging.info(f"Saved processed sequence for {row[id_column]} to {output_path}")
        else:
            logging.warning(f"Skipping sequence {row[id_column]} due to failed structure prediction.")

    return processed_df

if __name__ == "__main__":
    try:
        ln_seq_col = 'miRseq'
        lncbase_df = load_data('dataset/ENCORI_miRNA_lncRNA.csv', ln_seq_col)

        # Process and save structures
        output_file = 'mirna_sequences.csv'
        processed_lncbase_df = process_data(lncbase_df, ln_seq_col, 'miRNAid', output_file)
        logging.info("All data processed and saved successfully.")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
