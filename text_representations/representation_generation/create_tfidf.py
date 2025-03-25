import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

ufiles_path = ''
pfiles_path = ''

def create_reps_optimized(tp):
    print(f"\n📁 Processing type: {tp}")
    
    files = os.listdir(pfiles_path)
    file_contents = []
    file_names = []

    print("📥 Reading files...")
    for fname in tqdm(files):
        if tp == 'uniprot':
            with open(os.path.join(ufiles_path, fname), 'r', encoding='utf-8') as f:
                text = f.read()
        elif tp == 'pubmed':
            with open(os.path.join(pfiles_path, fname), 'r', encoding='utf-8') as f:
                text = f.read()
        elif tp == 'uniprotpubmed':
            with open(os.path.join(ufiles_path, fname), 'r', encoding='utf-8') as fu, \
                 open(os.path.join(pfiles_path, fname), 'r', encoding='utf-8') as fp:
                text = fu.read() + fp.read()
        else:
            raise ValueError(f"Unknown type: {tp}")

        file_contents.append(text)
        file_names.append(fname[:-4])  # Remove ".txt" or extension

    print("🧠 Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(use_idf=True, max_features=50000)  # optional: limit features
    tfidf_matrix = vectorizer.fit_transform(file_contents)  # sparse matrix

    print("💾 Saving TF-IDF matrix to CSV...")
    dense_matrix = tfidf_matrix.todense()
    df_tfidf = pd.DataFrame(dense_matrix, columns=vectorizer.get_feature_names_out())
    df_tfidf.insert(0, "Entry", file_names)

    output_dir = os.path.join(os.getcwd(), 'tfidf_representations')
    os.makedirs(output_dir, exist_ok=True)
    df_tfidf.to_csv(os.path.join(output_dir, f'{tp}_tfidf_vectors.csv'), index=False)

    # PCA alternatif: TruncatedSVD
    print("📉 Performing TruncatedSVD (PCA-like)...")
    for n_comp in [256, 512, 1024, 2048]:
        if n_comp < tfidf_matrix.shape[0] and n_comp < tfidf_matrix.shape[1]:
            print(f"🔧 Reducing to {n_comp} components...")
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            reduced = svd.fit_transform(tfidf_matrix)
            df_reduced = pd.DataFrame(reduced)
            df_reduced.insert(0, "Entry", file_names)
            df_reduced.to_csv(os.path.join(output_dir, f'{tp}_tfidf_vectors_svd{n_comp}.csv'), index=False)
        else:
            print(f"⚠️ Skipping SVD-{n_comp} (insufficient data)")

    print(f"✅ Done: {tp}\n")


def main():
    create_reps_optimized("uniprot")
    create_reps_optimized("pubmed")
    create_reps_optimized("uniprotpubmed")

