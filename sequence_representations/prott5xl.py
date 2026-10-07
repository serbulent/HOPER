"""ProtT5-XL per-protein sequence representations.

    conda activate prott5xl
    python sequence_representations/prott5xl.py --input proteins.csv --output prott5_bfd.csv
    python sequence_representations/prott5xl.py --input proteins.fasta --output prott5_bfd.csv --model uniref50

Input: a CSV with an identifier column (``Entry`` or ``ID``) and a ``Sequence`` column, or a FASTA file.
Output: a multi-column CSV (``Entry``, 0..1023) - the format used by the other HOPER modules.

Each protein vector is the mean of the encoder's per-residue embeddings (the final </s> token excluded);
residues are space-separated and the rare amino acids U, Z, O, B are mapped to X, as ProtT5 expects.

Models
------
- ``bfd`` (default): Rostlab/prot_t5_xl_bfd. This is the model behind the sequence vectors shipped in the
  example data (data/hoper_sequence_representations/T5_UNIPROT_HUMAN.csv; cosine similarity ~0.9999).
- ``uniref50``: Rostlab/prot_t5_xl_uniref50.

Only the encoder is needed, so on first use the encoder weights (~4.8 GB) are fetched from the published
safetensors checkpoint with HTTP range requests and cached in ``--model_dir`` (default:
sequence_representations/models, or $HOPER_MODEL_DIR). Loading needs ~6 GB of RAM instead of the
11.3 GB full checkpoint, which cannot be memory-mapped on machines with less RAM+swap than that.
"""
import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# repo, revision for config/tokenizer, revision that contains model.safetensors (pinned commits)
MODELS = {
    "bfd": ("Rostlab/prot_t5_xl_bfd",
            "7ae1d5c1d148d6c65c7e294cc72807e5b454fdb7", "30f629a340a36dbae4a215042a2438a94486b680"),
    "uniref50": ("Rostlab/prot_t5_xl_uniref50",
                 "973be27c52ee6474de9c945952a8008aeb2a1a73", "ecd1cb1a104eb6cec3fe0a04055ba6fbe2634b26"),
}
SIDE_FILES = ["config.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"]


# ----------------------------------------------------------------------------
# Model download (encoder only)
# ----------------------------------------------------------------------------
def _hf_url(repo, revision, filename):
    return "https://huggingface.co/{}/resolve/{}/{}".format(repo, revision, filename)


def _urlopen(url, byte_range=None, retries=5):
    headers = {"Range": "bytes={}-{}".format(*byte_range)} if byte_range else {}
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def prepare_encoder(model, model_dir):
    """Return a local directory holding the ProtT5 encoder (downloaded on first use)."""
    repo, config_rev, weights_rev = MODELS[model]
    target = os.path.join(model_dir, repo.split("/")[1] + "-encoder")
    weights = os.path.join(target, "model.safetensors")
    if os.path.isfile(weights) and all(os.path.isfile(os.path.join(target, f)) for f in SIDE_FILES):
        return target
    os.makedirs(target, exist_ok=True)
    for filename in SIDE_FILES:
        with _urlopen(_hf_url(repo, config_rev, filename)) as resp, open(os.path.join(target, filename), "wb") as out:
            shutil.copyfileobj(resp, out)

    url = _hf_url(repo, weights_rev, "model.safetensors")
    with _urlopen(url, (0, 7)) as resp:
        header_len = struct.unpack("<Q", resp.read())[0]
    with _urlopen(url, (8, 8 + header_len - 1)) as resp:
        header = json.loads(resp.read())
    header.pop("__metadata__", None)
    data_start = 8 + header_len

    # The shared token embedding is tied; checkpoints store it under one of these names.
    shared_key = next(k for k in ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight")
                      if k in header)
    sources = {"shared.weight": shared_key}
    sources.update({k: k for k in header if k.startswith("encoder.") and k != "encoder.embed_tokens.weight"})

    new_header, offset = {}, 0
    for name in sorted(sources):
        begin, end = header[sources[name]]["data_offsets"]
        new_header[name] = {"dtype": header[sources[name]]["dtype"], "shape": header[sources[name]]["shape"],
                            "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    header_bytes = json.dumps(new_header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)

    print("Downloading the {} encoder ({:.1f} GB) to {} ...".format(repo, offset / 1e9, target), flush=True)
    tmp = weights + ".part"
    with open(tmp, "wb") as out, tqdm(total=offset, unit="B", unit_scale=True) as bar:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for name in sorted(sources):
            begin, end = header[sources[name]]["data_offsets"]
            with _urlopen(url, (data_start + begin, data_start + end - 1)) as resp:
                while True:
                    chunk = resp.read(16 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))
    if os.path.getsize(tmp) != 8 + len(header_bytes) + offset:
        raise RuntimeError("Incomplete download of {}; delete {} and retry.".format(url, tmp))
    os.replace(tmp, weights)
    return target


def load_model(model, model_dir, device):
    from transformers import T5EncoderModel, T5Tokenizer
    path = prepare_encoder(model, model_dir)
    tokenizer = T5Tokenizer.from_pretrained(path, do_lower_case=False, legacy=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    encoder = T5EncoderModel.from_pretrained(path, torch_dtype=dtype).to(device).eval()
    return tokenizer, encoder


# ----------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------
def read_sequences(path, id_column=None):
    """Read a FASTA file or a CSV with an id column (Entry/ID) and a Sequence column."""
    if path.lower().endswith((".fa", ".fasta", ".faa", ".fas")):
        entries, seqs, current = [], [], None
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith(">"):
                    current = line[1:].split()[0]
                    entries.append(current)
                    seqs.append("")
                elif line and current is not None:
                    seqs[-1] += line
        return pd.DataFrame({"Entry": entries, "Sequence": seqs})
    df = pd.read_csv(path)
    id_column = id_column or next((c for c in ("Entry", "ID") if c in df.columns), None)
    if id_column is None or "Sequence" not in df.columns:
        raise SystemExit("CSV input needs an id column (Entry or ID, or --id_column) and a Sequence column.")
    return df[[id_column, "Sequence"]].rename(columns={id_column: "Entry"})


def prepare_sequence(sequence):
    return " ".join(re.sub(r"[UZOB]", "X", sequence.strip().upper()))


def embed_dataframe(df, tokenizer, encoder, device, batch_size=8, max_length=2048, output_csv=None):
    """Mean-pooled ProtT5 vectors for df[Entry, Sequence]. Written batch by batch when output_csv is given."""
    df = df.dropna(subset=["Sequence"]).copy()
    df["Sequence"] = df["Sequence"].astype(str).str.strip()
    too_long = df["Sequence"].str.len() > max_length
    for entry in df.loc[too_long, "Entry"]:
        print("Skipping {}: longer than {} residues (--max_length)".format(entry, max_length), file=sys.stderr)
    df = df.loc[~too_long]
    df = df.iloc[np.argsort(df["Sequence"].str.len().values, kind="stable")]  # less padding per batch

    results, header_written = [], False
    for start in tqdm(range(0, len(df), batch_size), desc="Embedding"):
        batch = df.iloc[start:start + batch_size]
        lengths = batch["Sequence"].str.len().tolist()
        inputs = tokenizer([prepare_sequence(s) for s in batch["Sequence"]], add_special_tokens=True,
                           padding="longest", return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = encoder(**inputs).last_hidden_state.float()
        vectors = np.stack([hidden[i, :lengths[i]].mean(dim=0).cpu().numpy() for i in range(len(lengths))])
        frame = pd.DataFrame(vectors)
        frame.insert(0, "Entry", batch["Entry"].values)
        if output_csv:
            frame.to_csv(output_csv, mode="a" if header_written else "w", header=not header_written, index=False)
            header_written = True
        else:
            results.append(frame)
    if output_csv:
        if not header_written:
            raise SystemExit("No sequences to embed.")
        return None
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="ProtT5-XL per-protein representations")
    parser.add_argument("--input", required=True, help="CSV (Entry/ID + Sequence columns) or FASTA file")
    parser.add_argument("--output", required=True, help="Output multi-column CSV")
    parser.add_argument("--model", choices=sorted(MODELS), default="bfd",
                        help="bfd (default, matches the example data) or uniref50")
    parser.add_argument("--id_column", default=None, help="Identifier column of a CSV input (default: Entry or ID)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048, help="Skip longer sequences (memory)")
    parser.add_argument("--model_dir", default=os.environ.get("HOPER_MODEL_DIR", os.path.join(MODULE_DIR, "models")))
    parser.add_argument("--device", default=os.environ.get("HOPER_DEVICE", "cpu"), help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    df = read_sequences(args.input, args.id_column)
    print("{} sequences, model {}, device {}".format(len(df), args.model, device), flush=True)
    tokenizer, encoder = load_model(args.model, args.model_dir, device)
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    embed_dataframe(df, tokenizer, encoder, device, args.batch_size, args.max_length, output_csv=args.output)
    print("Saved " + args.output)


if __name__ == "__main__":
    main()
