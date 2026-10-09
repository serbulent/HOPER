"""Reading and aligning multi-column representation CSVs (Entry, 0, 1, ...) for the autoencoders."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_representation(path, scale=False):
    """Return a float32 DataFrame indexed by Entry (leading pandas index columns are ignored).
    With scale=True every dimension is standardised over the whole file."""
    df = pd.read_csv(path)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed:")]]
    if "Entry" not in df.columns:
        raise SystemExit("{}: an 'Entry' column is required.".format(path))
    df = df.drop_duplicates("Entry").set_index("Entry").astype(np.float32)
    if scale:
        df[:] = StandardScaler().fit_transform(df.values).astype(np.float32)
    print("{}: {} proteins x {} dimensions".format(path, df.shape[0], df.shape[1]))
    return df


def standardisation_factors(df):
    """shift, scale such that (x + shift) * scale standardises every dimension of df (as StandardScaler does)."""
    values = df.values.astype(np.float64)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return -values.mean(axis=0), 1.0 / std


def apply_factors(df, shift, scale):
    out = df.copy()
    out[:] = ((df.values.astype(np.float64) + shift) * scale).astype(np.float32)
    return out


def factor_paths(weights_path):
    """Files that store the sequence standardisation factors next to transfer-model weights."""
    return weights_path + ".shift_factors.txt", weights_path + ".scaling_factors.txt"


def write_factors(path, values):
    np.savetxt(path, np.asarray(values, dtype=np.float64).reshape(-1, 1))


def read_factors(path):
    with open(path) as handle:
        return np.array([float(v) for line in handle for v in line.split()])


def align(*frames):
    """Keep the proteins present in every frame (order of the first frame); return entries and arrays."""
    common = frames[0].index
    for frame in frames[1:]:
        common = common[common.isin(frame.index)]
    if len(common) == 0:
        raise SystemExit("The representation files have no protein (Entry) in common.")
    print("Proteins present in all inputs: {}".format(len(common)))
    return list(common), [frame.loc[common].values for frame in frames]


def to_multi_col(entries, vectors):
    """Entries + 2-D array -> multi-column DataFrame (Entry, 0..n-1)."""
    out = pd.DataFrame(np.asarray(vectors))
    out.insert(0, "Entry", entries)
    return out
