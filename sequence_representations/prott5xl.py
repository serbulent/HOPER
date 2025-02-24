import numpy as np
import pandas as pd
import torch
import gc
from tqdm import tqdm
#from bio_embeddings.embed import ProtTransT5BFDEmbedder
from transformers import AutoModel, AutoTokenizer
from transformers import T5EncoderModel, T5Tokenizer

def initialize_t5_embedder():
    """
    Initialize the ProtTransT5BFDEmbedder on the available CUDA device.

    This function checks for CUDA availability, prints the current CUDA device details,
    and initializes the ProtTransT5BFDEmbedder with half-precision model settings.

    Returns:
        ProtTransT5BFDEmbedder: An initialized T5 embedder object ready for sequence embedding.
    """
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

    #t5_embedder = ProtTransT5BFDEmbedder(device=torch.cuda.current_device(), haf_precision_model=True)
    #return t5_embedder


def convert_dataframe_to_multi_col(representation_dataframe, id_column):
    """
    Convert a DataFrame with vector representations into a multi-column format.

    This function takes a DataFrame containing IDs and vector representations
    (stored as a single column of lists) and splits the vector column into multiple columns.

    Args:
        representation_dataframe (pd.DataFrame): DataFrame containing IDs and Vectors.
        id_column (str): Name of the ID column (e.g., 'Entry', 'PDB_ID').

    Returns:
        pd.DataFrame: A new DataFrame where each dimension of the vector is a separate column,
        with the ID column retained.

    Example:
        Input DataFrame:
            Entry    Vector
            A0A0B4  [0.1, 0.2, 0.3]

        Output DataFrame:
            Entry    0    1    2
            A0A0B4  0.1  0.2  0.3
    """
    entry = pd.DataFrame(representation_dataframe[id_column])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    multi_col_representation_vector = pd.merge(left=entry, right=vector, left_index=True, right_index=True)
    return multi_col_representation_vector


def load_uniprot_metadata(uniprot_metadata_file_path):
    """
    Load UniProt metadata from a specified file path.

    This function reads a tab-delimited file containing UniProt metadata and loads
    it into a Pandas DataFrame. It ensures that only relevant columns are retained.

    Args:
        uniprot_metadata_file_path (str): Path to the UniProt metadata file.

    Returns:
        pd.DataFrame: A DataFrame containing UniProt metadata with predefined columns.

    Expected Columns:
        - Entry
        - Entry name
        - Status
        - Protein names
        - Gene names
        - Organism
        - Length
        - Annotation

    Example:
        File Format:
            Entry	Entry name	Status	...
            P12345	MYPROTEIN_HUMAN	Reviewed	...
    """
    uniprot_vars = ['Entry', 'Entry name', 'Status', 'Protein names', 'Gene names', 'Organism', 'Length', 'Annotation']
    uniprot_df = pd.read_csv(uniprot_metadata_file_path, sep='\t', usecols=uniprot_vars)
    return uniprot_df


def process_protein_sequences(protein_sequence_records, uniprot_df):
    """
    Standardize protein sequence records with UniProt entry IDs.

    This function maps protein sequences to their corresponding UniProt entry IDs
    using the provided UniProt metadata. It creates a standardized DataFrame
    that includes UniProt entry IDs and sequences.

    Args:
        protein_sequence_records (pd.DataFrame): DataFrame with columns:
            - SwissProtEntryName: The UniProt entry name (e.g., MYPROTEIN_HUMAN).
            - Sequence: The protein sequence.
        uniprot_df (pd.DataFrame): UniProt metadata DataFrame.

    Returns:
        pd.DataFrame: A standardized DataFrame with columns:
            - Entry: The UniProt entry ID.
            - Sequence: The protein sequence.

    Note:
        If no matching UniProt entry is found, the function will raise an error.
    """
    protein_sequence_records_standard_df = pd.DataFrame({'Entry': [], 'Sequence': []}, dtype='str')

    for index, record in tqdm(protein_sequence_records.iterrows(), total=len(protein_sequence_records)):
        entry = uniprot_df[uniprot_df['Entry name'] == record['SwissProtEntryName']]['Entry'].item()
        protein_sequence_records_standard_df = pd.concat([protein_sequence_records_standard_df, 
            pd.DataFrame({'Entry': [entry], 'Sequence': [record['Sequence']]})], ignore_index=True)

    return protein_sequence_records_standard_df


def generate_embeddings(protein_sequence_records_standard_df, t5_embedder):
    """
    Generate ProtTransT5 embeddings for protein sequences.

    This function computes vector embeddings for protein sequences using the
    ProtTransT5BFDEmbedder. Each sequence's embedding is averaged across all positions
    to produce a single vector representation.

    Args:
        protein_sequence_records_standard_df (pd.DataFrame): DataFrame with columns:
            - Entry: The UniProt entry ID.
            - Sequence: The protein sequence.
        t5_embedder (ProtTransT5BFDEmbedder): An initialized T5 embedder object.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - Entry: The UniProt entry ID.
            - Vector: The computed embedding (as a list of floats).

    Note:
        Sequences longer than 2048 residues are skipped due to input size constraints.
    """
    
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    
    prottrans_t5_representation_df = pd.DataFrame({'Entry': [], 'Vector': []}, dtype='object')
    
    gc.collect()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    
    ids = tokenizer.batch_encode_plus(protein_sequence_records_standard_df, add_special_tokens=True, padding=True)
    
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    for index, record in tqdm(protein_sequence_records_standard_df.iterrows(),
                              total=len(protein_sequence_records_standard_df)):
        seq = record['Sequence']
        if len(seq) < 2048:

            with torch.no_grad():
                embedding = model(input_ids=input_ids,attention_mask=attention_mask)
            
            protein_embd = embedding.last_hidden_state[0,:7].cpu().numpy().mean(axis=0)
            #print(protein_embd.shape)
            prottrans_t5_representation_df = pd.concat([prottrans_t5_representation_df, 
                pd.DataFrame({'Entry': [record['Entry']], 'Vector': [protein_embd]})], ignore_index=True)
            del protein_embd
            torch.cuda.empty_cache()
            gc.collect()

    return prottrans_t5_representation_df


def main():
    """
    Main function to run the embedding generation pipeline.

    This function prompts the user to provide file paths for:
        1. UniProt metadata.
        2. Protein sequence records.
        3. Output file for the computed embeddings.

    Steps:
        1. Load UniProt metadata.
        2. Load protein sequence records.
        3. Initialize the T5 embedder.
        4. Process protein sequences to map them to UniProt entries.
        5. Generate embeddings for the processed sequences.
        6. Convert embeddings to multi-column format.
        7. Save the multi-column embeddings to the specified output file.

    Example:
        Input:
            - UniProt metadata file: uniprot_metadata.tsv
            - Protein sequence file: sequences.csv
            - Output file: embeddings.csv

        Output:
            The embeddings.csv file will contain the computed embeddings in multi-column format.
    """
    # Get file paths from user
    uniprot_metadata_file_path = input("Enter the path to the UniProt metadata file: ")
    protein_sequence_records_path = input("Enter the path to the protein sequence records file: ")
    output_file_path = input("Enter the output file path for the embeddings: ")
    
    # Load UniProt metadata
    uniprot_df = load_uniprot_metadata(uniprot_metadata_file_path)

    # Load protein sequence records
    protein_sequence_records = pd.read_csv(protein_sequence_records_path)

    # Initialize T5 embedder
    t5_embedder = initialize_t5_embedder()

    # Process protein sequences
    protein_sequence_records_standard_df = process_protein_sequences(protein_sequence_records, uniprot_df)

    # Generate embeddings
    prottrans_t5_representation_df = generate_embeddings(protein_sequence_records_standard_df, t5_embedder)
    #print(prottrans_t5_representation_df.head())
    # Convert to multi-column format
    prottrans_t5_representation_multi_col = convert_dataframe_to_multi_col(
        prottrans_t5_representation_df, id_column='Entry')

    # Save results
    prottrans_t5_representation_multi_col.to_csv(output_file_path, index=False)
    print(f"Embeddings saved to {output_file_path}")


if __name__ == "__main__":
    main()

