import os

import pandas as pd
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import rep_paths

ufiles_path = ''
pfiles_path = ''

SVD_COMPONENTS = [256, 512, 1024, 2048]


def create_reps_optimized(tp):
    print(f"\nProcessing type: {tp}")
    files = sorted(os.listdir(pfiles_path))
    file_contents = []
    file_names = []

    print("Reading files...")
    for fname in tqdm(files):
        if tp == 'uniprot':
            text = rep_paths.read_text(ufiles_path, fname)
        elif tp == 'pubmed':
            text = rep_paths.read_text(pfiles_path, fname)
        elif tp == 'uniprotpubmed':
            text = rep_paths.read_text(ufiles_path, fname) + rep_paths.read_text(pfiles_path, fname)
        else:
            raise ValueError(f"Unknown type: {tp}")
        file_contents.append(text)
        file_names.append(os.path.splitext(fname)[0])

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(use_idf=True, max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(file_contents)  # sparse; never densified in full
    output_dir = rep_paths.output_dir("tfidf")

    # The full matrix is stored sparse (a dense CSV of 20k proteins x 50k terms needs ~8 GB of RAM).
    scipy.sparse.save_npz(os.path.join(output_dir, f'{tp}_tfidf_vectors.npz'), tfidf_matrix)
    pd.DataFrame({"Entry": file_names}).to_csv(os.path.join(output_dir, f'{tp}_tfidf_entries.csv'), index=False)
    pd.Series(vectorizer.get_feature_names_out()).to_csv(
        os.path.join(output_dir, f'{tp}_tfidf_vocabulary.csv'), index=False, header=["term"])
    if os.environ.get("HOPER_TFIDF_DENSE_CSV") == "1":
        df_tfidf = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, columns=vectorizer.get_feature_names_out())
        df_tfidf.insert(0, "Entry", file_names)
        df_tfidf.sparse.to_dense().to_csv(os.path.join(output_dir, f'{tp}_tfidf_vectors.csv'), index=False)

    print("Performing TruncatedSVD...")
    for n_comp in SVD_COMPONENTS:
        if n_comp < tfidf_matrix.shape[0] and n_comp < tfidf_matrix.shape[1]:
            print(f"Reducing to {n_comp} components...")
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            reduced = svd.fit_transform(tfidf_matrix)
            df_reduced = pd.DataFrame(reduced)
            df_reduced.insert(0, "Entry", file_names)
            df_reduced.to_csv(os.path.join(output_dir, f'{tp}_tfidf_vectors_svd{n_comp}.csv'), index=False)
            del svd, reduced, df_reduced
        else:
            print(f"Skipping SVD-{n_comp} (insufficient data)")

    print(f"Done: {tp}\n")


def main():
    create_reps_optimized("uniprot")
    create_reps_optimized("pubmed")
    create_reps_optimized("uniprotpubmed")
