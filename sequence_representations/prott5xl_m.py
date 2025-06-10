import numpy as np
import pandas as pd
import torch
import gc
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from Bio import SeqIO


def preprocess_sequence(sequence):
    return ' '.join(list(sequence.strip()))

def initialize_t5_model_and_tokenizer():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50",
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

def convert_dataframe_to_multi_col(representation_dataframe, id_column):
    entry = pd.DataFrame(representation_dataframe[id_column])
    vector = pd.DataFrame(list(representation_dataframe['Sequence']))
    multi_col_representation_vector = pd.merge(
        left=entry, right=vector, left_index=True, right_index=True
    )
    return multi_col_representation_vector

def _embed_dataframe(df, tokenizer, model, device, batch_size, output_csv=None):
    results = []
    num_sequences = len(df)

    for i in tqdm(range(0, num_sequences, batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        valid_rows = batch_df[batch_df['Sequence'].apply(lambda s: isinstance(s, str) and len(s) < 2048)]

        if valid_rows.empty:
            for idx, row in batch_df.iterrows():
                print(f"Skipping sequence with ID {row['Entry']} due to excessive length.")
            continue

        sequences = [preprocess_sequence(seq) for seq in valid_rows['Sequence'].tolist()]
        identifiers = valid_rows['Entry'].tolist()

        inputs = tokenizer(
            sequences,
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        last_hidden = output.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings = embeddings.cpu().numpy().astype(np.float32)

        batch_results = [{'Entry': ident, 'Sequence': emb} for ident, emb in zip(identifiers, embeddings)]
        results.extend(batch_results)

        del inputs, output, last_hidden, mask, embeddings, batch_results
        torch.cuda.empty_cache()
        gc.collect()

        if output_csv:
            temp_df = convert_dataframe_to_multi_col(pd.DataFrame(results), id_column='Entry')
            mode = 'a' if i > 0 else 'w'
            header = (i == 0)
            temp_df.to_csv(output_csv, mode=mode, header=header, index=False)
            results = []

    if not output_csv:
        embeddings_df = pd.DataFrame(results)
        embeddings_multi_col_df = convert_dataframe_to_multi_col(embeddings_df, id_column='Entry')
        return embeddings_multi_col_df
    else:
        return None

def embed_sequence(sequence):
    tokenizer, model, device = initialize_t5_model_and_tokenizer()
    return generate_embedding_for_sequence(sequence, tokenizer, model, device)

def embed_fasta(file_path, output_file=None, batch_size=32):
    records = list(SeqIO.parse(file_path, "fasta"))
    data = [{'Entry': record.id, 'Sequence': str(record.seq)} for record in records]
    df = pd.DataFrame(data)

    tokenizer, model, device = initialize_t5_model_and_tokenizer()
    return _embed_dataframe(df, tokenizer, model, device, batch_size, output_csv=output_file)

def main():
    import os
    file_path = "/media/DATA2/testuser2/prott5_sequence.csv"
    df = pd.read_csv(file_path, usecols=['Entry', 'Sequence'])
    tokenizer, model, device = initialize_t5_model_and_tokenizer()
    batch_size = 4
    file_name = os.path.splitext(os.path.basename(file_path))[0] + "_Prott5xl.csv"
    output_path = os.path.join(os.getcwd(), file_name)
    _embed_dataframe(df, tokenizer, model, device, batch_size, output_csv=output_path)
    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    main()
