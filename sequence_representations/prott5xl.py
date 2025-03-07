import numpy as np
import pandas as pd
import torch
import gc
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from Bio import SeqIO

def initialize_t5_model_and_tokenizer():
    """
    Initialize the T5 model and tokenizer for ProtT5.
    
    Returns:
        tokenizer: The ProtT5 tokenizer.
        model: The ProtT5 encoder model on the appropriate device.
        device: The torch device (GPU if available, else CPU).
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

def generate_embedding_for_sequence(sequence, tokenizer, model, device):
    """
    Generate a ProtT5 embedding for a single protein sequence.
    
    For individual sequence embedding, a simple tokenization and model inference is performed.
    In batched mode (used in FASTA input processing), mean pooling with attention masks is applied
    to aggregate token-level embeddings into a fixed-size vector, excluding padded tokens.
    
    Args:
        sequence (str): The protein sequence.
        tokenizer: ProtT5 tokenizer.
        model: ProtT5 model.
        device: Torch device.
    
    Returns:
        np.ndarray or None: Mean-pooled embedding vector, or None if the sequence is too long.
    """
    if len(sequence) >= 2048:
        print("Sequence is too long, skipping.")
        return None

    # Tokenize the input sequence
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)
    
    # For a single sequence, we simply average all token embeddings.
    # In batched mode, mask-based pooling is applied to avoid the influence of padding tokens.
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def convert_dataframe_to_multi_col(representation_dataframe, id_column):
    """
    Convert a DataFrame with vector representations into a multi-column format.
    
    Each element in the 'Vector' column (assumed to be a list/array) is split into
    its own column while retaining the identifier column.
    
    Args:
        representation_dataframe (pd.DataFrame): DataFrame with columns for IDs and 'Vector'.
        id_column (str): Name of the identifier column.
    
    Returns:
        pd.DataFrame: DataFrame where each dimension of the vector is a separate column.
    """
    entry = pd.DataFrame(representation_dataframe[id_column])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    multi_col_representation_vector = pd.merge(left=entry, right=vector, left_index=True, right_index=True)
    return multi_col_representation_vector

def _embed_dataframe(df, tokenizer, model, device, batch_size):
    """
    Helper function to generate embeddings for a DataFrame containing protein sequences.
    
    This function processes the DataFrame in batches. For each batch, the sequences are
    tokenized and passed through the model. Mean pooling is applied using the attention mask 
    to ensure that only non-padded tokens contribute to the embedding. This produces a single
    fixed-size vector per sequence.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'ID' and 'Sequence'.
        tokenizer: ProtT5 tokenizer.
        model: ProtT5 model.
        device: Torch device.
        batch_size (int): Number of sequences to process in one batch.
    
    Returns:
        pd.DataFrame: A DataFrame with each row containing an 'ID' and a 'Vector' (embedding),
                      later converted to multi-column format.
    """
    results = []
    num_sequences = len(df)
    for i in tqdm(range(0, num_sequences, batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        # Filter out sequences that are too long
        valid_rows = batch_df[batch_df['Sequence'].apply(lambda s: len(s) < 2048)]
        if valid_rows.empty:
            for idx, row in batch_df.iterrows():
                print(f"Skipping sequence with ID {row['ID']} due to excessive length.")
            continue

        sequences = valid_rows['Sequence'].tolist()
        identifiers = valid_rows['ID'].tolist()

        # Batch tokenize the sequences using the tokenizer's batching method.
        inputs = tokenizer(sequences, return_tensors="pt", add_special_tokens=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
        
        # Mean pooling using the attention mask to avoid the influence of padding tokens.
        last_hidden = output.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        mask = inputs['attention_mask'].unsqueeze(-1)  # shape: (batch_size, seq_len, 1)
        embeddings = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # shape: (batch_size, hidden_size)
        embeddings = embeddings.cpu().numpy()

        for ident, emb in zip(identifiers, embeddings):
            results.append({'ID': ident, 'Vector': emb})
        
        torch.cuda.empty_cache()
        gc.collect()
    
    embeddings_df = pd.DataFrame(results)
    embeddings_multi_col_df = convert_dataframe_to_multi_col(embeddings_df, id_column='ID')
    return embeddings_multi_col_df

def embed_sequence(sequence):
    """
    Generate an embedding for a single protein sequence.
    
    Args:
        sequence (str): Protein sequence.
    
    Returns:
        np.ndarray or None: The embedding vector, or None if the sequence is too long.
    
    Note:
        For single sequence embedding, a simple mean pooling is applied.
        In batch processing (FASTA inputs), mean pooling with an attention mask is used to handle padded tokens.
    """
    tokenizer, model, device = initialize_t5_model_and_tokenizer()
    embedding = generate_embedding_for_sequence(sequence, tokenizer, model, device)
    return embedding

def embed_fasta(file_path, output_file=None, batch_size=32):
    """
    Generate embeddings for protein sequences contained in a FASTA file using batching.
    
    The FASTA file is expected to have sequence headers containing the sequence ID.
    Sequences longer than 2048 residues are skipped. The function processes sequences in batches,
    using mask-based mean pooling to aggregate token-level embeddings into a single vector per sequence.
    This ensures that padding tokens do not affect the final embedding.
    
    Args:
        file_path (str): Path to the input FASTA file.
        output_file (str, optional): If provided, the multi-column embeddings are saved to this CSV file.
        batch_size (int): Number of sequences to process in one batch.
    
    Returns:
        pd.DataFrame: DataFrame of embeddings in multi-column format.
    """
    # Parse the FASTA file into a DataFrame with columns 'ID' and 'Sequence'
    records = list(SeqIO.parse(file_path, "fasta"))
    data = [{'ID': record.id, 'Sequence': str(record.seq)} for record in records]
    df = pd.DataFrame(data)
    
    tokenizer, model, device = initialize_t5_model_and_tokenizer()
    embeddings_multi_col_df = _embed_dataframe(df, tokenizer, model, device, batch_size)
    
    if output_file is not None:
        embeddings_multi_col_df.to_csv(output_file, index=False)
    
    return embeddings_multi_col_df
