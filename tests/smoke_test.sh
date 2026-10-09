#!/usr/bin/env bash
# End-to-end smoke test of the README workflow on small example inputs.
#   bash tests/smoke_test.sh            # all steps
#   bash tests/smoke_test.sh --quick    # skip the model downloads (ProtT5 ~4.8 GB, BioBERT) and preprocessing
# Prerequisites: bash create_env.sh && bash download_data.sh [--uniprot]
# Each step checks the produced files, not only the exit code. Prints a summary; exit code 1 if any step failed.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
TMP="$ROOT/tests/tmp"
LOG="$TMP/logs"
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
rm -rf "$TMP"; mkdir -p "$LOG"
export HOPER_DEVICE=${HOPER_DEVICE:-cpu}

command -v conda >/dev/null 2>&1 || { echo "conda not found: activate your conda installation first."; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
NAMES=(); RESULTS=()
step() { # <name> <function>   (set SMOKE_ONLY="t_ppi t_fuse ..." to run selected steps only)
  if [ -n "${SMOKE_ONLY:-}" ] && [[ " $SMOKE_ONLY " != *" $2 "* ]]; then return; fi
  echo; echo "======== $1"
  local start=$SECONDS
  if "$2" > "$LOG/$2.log" 2>&1; then RESULTS+=("PASS ($((SECONDS - start))s)"); else RESULTS+=("FAIL ($((SECONDS - start))s) -> tests/tmp/logs/$2.log"); tail -15 "$LOG/$2.log"; fi
  NAMES+=("$1"); echo "${RESULTS[-1]}"
}
launcher() { conda run --no-capture-output -n hoper python Hoper_representation_generetor_main.py "$1"; }
inenv() { local env=$1; shift; conda run --no-capture-output -n "$env" "$@"; }
pycheck() { inenv HoloProtRep-AE python -c "$1"; }

config() { # <name> <yaml body>  -> writes tests/tmp/<name>.yaml
  printf 'parameters:\n%s\n' "$2" > "$TMP/$1.yaml"; echo "$TMP/$1.yaml"
}

t_ppi() {
  rm -f data/Node2vec_d_10_p_0.25_q_0.25.pkl data/HOPE_d_5_beta_0.00390625.pkl
  launcher "$(config ppi '    choice_of_module: [PPI]
    choice_of_representation_name: [Node2vec,HOPE]
    interaction_data_path: [./data/hoper_PPI/PPI_example_data/example.edgelist]
    protein_id_list: [./data/hoper_PPI/PPI_example_data/proteins_id.csv]
    is_directed: false
    node2vec_module: {parameter_selection: {d: [10], p: [0.25], q: [0.25]}}
    HOPE_module: {parameter_selection: {d: [5], beta: [0.00390625]}}')" | tee "$TMP/ppi.out" || return 1
  # undirected graph: every distinct undirected edge must appear in both directions
  local expected
  expected=$(awk '{a=$1; b=$2; if (a>b) {t=a; a=b; b=t}; e[a" "b]=1} END {n=0; for (k in e) n++; print 2*n}' \
             data/hoper_PPI/PPI_example_data/example.edgelist)
  grep -q "num edges: $expected" "$TMP/ppi.out" || { echo "expected $expected directed edges"; return 1; }
  inenv hoper_PPI python -c "
import pickle
for f in ['data/Node2vec_d_10_p_0.25_q_0.25.pkl', 'data/HOPE_d_5_beta_0.00390625.pkl']:
    df = pickle.load(open(f, 'rb')); assert df.shape == (5, 2), df.shape; print(f, df.shape)"
}

t_sequence() {
  # ProtT5-XL (bfd) vectors must match the authors' vectors shipped in the example data
  local out="$TMP/prott5_bfd.csv"
  launcher "$(config seq "    choice_of_module: [sequence]
    sequence_module: {input_path: ./sequence_representations/example_sequences.fasta, output_path: $out, model: bfd, batch_size: 4}")" || return 1
  pycheck "
import numpy as np, pandas as pd
new = pd.read_csv('$out').set_index('Entry')
ref = pd.read_csv('data/hoper_sequence_representations/T5_UNIPROT_HUMAN.csv')
ref = ref.loc[:, [c for c in ref.columns if not str(c).startswith('Unnamed')]].set_index('Entry')
assert new.shape == (3, 1024), new.shape
for e in new.index:
    a, b = new.loc[e].values.astype(float), ref.loc[e].values.astype(float)
    c = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)); print(e, 'cosine vs T5_UNIPROT_HUMAN', round(c, 5)); assert c > 0.999"
}

t_fuse() {
  local out=data/node2vec_modal_rep_ae_binary_fused_representations_dataframe_multi_col.csv; rm -f "$out"
  launcher "$(config fuse '    choice_of_module: [fuse_representations]
    representation_files: [./data/hoper_case_study_example_data/representation_files/node2vec_d_50_p_0.5_q_0.25_multi_col.csv,./data/hoper_case_study_example_data/representation_files/multi_modal_rep_ae_multi_col_256.csv]
    min_fold_number: 2
    representation_names: [node2vec,modal_rep_ae]')" || return 1
  pycheck "
import pandas as pd; d = pd.read_csv('$out', nrows=5); print(d.shape); assert d.shape[1] == 1 + 50 + 384"
}

make_text_subset() { # <n> -> tests/tmp/text_<n>/{uniprot,pubmed}: proteins that also have a Node2vec PPI vector
  local d="$TMP/text_$1"; mkdir -p "$d/uniprot" "$d/pubmed"
  local src=text_representations/representation_generation/data
  cut -d, -f1 ./data/hoper_case_study_example_data/representation_files/node2vec_d_50_p_0.5_q_0.25_multi_col.csv \
    | tail -n +2 | sed 's/$/.txt/' > "$TMP/ppi_files.txt"
  ls "$src/pubmed" | grep -Fxf "$TMP/ppi_files.txt" | head -n "$1" \
    | while read -r f; do cp "$src/uniprot/$f" "$d/uniprot/"; cp "$src/pubmed/$f" "$d/pubmed/"; done
  echo "$d"
}

t_text_tfidf() {
  local d; d=$(make_text_subset 300)
  local out=text_representations/representation_generation/tfidf_representations
  rm -f "$out"/*_svd256.csv
  launcher "$(config tfidf "    choice_of_module: [text]
    choice_of_process: [generate]
    generate_module: {choice_of_representation_type: [tfidf], uniprot_files_path: [$d/uniprot/], pubmed_files_path: [$d/pubmed/], model_download: n}")" || return 1
  pycheck "
import pandas as pd
for tp in ['uniprot', 'pubmed', 'uniprotpubmed']:
    d = pd.read_csv('$out/' + tp + '_tfidf_vectors_svd256.csv'); print(tp, d.shape); assert d.shape == (300, 257)"
}

t_text_biobert() {
  local d; d=$(make_text_subset 20)
  local out=text_representations/representation_generation/biobert_representations
  rm -f "$out"/*.csv
  launcher "$(config biobert "    choice_of_module: [text]
    choice_of_process: [generate]
    generate_module: {choice_of_representation_type: [biobert], uniprot_files_path: [$d/uniprot/], pubmed_files_path: [$d/pubmed/], model_download: n}")" || return 1
  pycheck "
import pandas as pd
for tp in ['uniprot', 'pubmed', 'uniprotpubmed']:
    d = pd.read_csv('$out/' + tp + '_biobert_embeddings_multi_col.csv'); print(tp, d.shape); assert d.shape == (20, 769)"
}

t_text_missing_model() {
  # BioSentVec without the 22 GB model must fail fast with a clear message (not a silent "model loaded")
  local d; d=$(make_text_subset 5)
  if launcher "$(config bsv "    choice_of_module: [text]
    choice_of_process: [generate]
    generate_module: {choice_of_representation_type: [biosentvec], uniprot_files_path: [$d/uniprot/], pubmed_files_path: [$d/pubmed/], model_download: n}")" > "$TMP/bsv.out" 2>&1; then
    echo "expected a failure without the model"; return 1
  fi
  cat "$TMP/bsv.out"; grep -q "BioSentVec model not found" "$TMP/bsv.out"
}

t_visualize() {
  local fig=text_representations/result_visualization/figures
  rm -f "$fig"/func_pred_*.png
  launcher "$(config vis '    choice_of_module: [text]
    choice_of_process: [visualize]
    visualize_module: {choice_of_visualization_type: [a], result_files_path: [./text_representations/result_visualization/result_files/results/]}')" || return 1
  ls -la "$fig"/func_pred_BP.png "$fig"/func_pred_CC.png "$fig"/func_pred_MF.png
}

t_preprocess() {
  [ -f uniprot_sprot.xml.gz ] || { echo "uniprot_sprot.xml.gz missing: run download_data.sh --uniprot"; return 1; }
  # first 300 Swiss-Prot entries, so the step finishes in seconds
  inenv hoper_preprocess python -c "
import gzip
out = gzip.open('$TMP/uniprot_mini.xml.gz', 'wt'); n = 0
with gzip.open('uniprot_sprot.xml.gz', 'rt') as f:
    for line in f:
        if line.startswith('<entry') and n == 300: break
        out.write(line); n += line.startswith('<entry')
out.write('</uniprot>\n'); out.close(); print('entries', n)" || return 1
  local pd=text_representations/preprocess/data
  rm -rf "$pd"/uniprot_subsections "$pd"/uniprot_par "$pd"/uniprot_dot "$pd"/uniprot_space "$pd"/human_pubmed_ids
  env -u HOPER_ENTREZ_EMAIL bash -c "conda run --no-capture-output -n hoper python Hoper_representation_generetor_main.py '$(config pre "    choice_of_module: [Preprocessing]
    uniprot_dir: $TMP/uniprot_mini.xml.gz")'" || return 1
  for d in uniprot_subsections uniprot_space human_pubmed_ids; do
    n=$(ls "$pd/$d" | wc -l); echo "$d: $n files"; [ "$n" -gt 0 ] || return 1
  done
}

t_simple_ae() {
  local out="$TMP/simple_ae"
  local fused=./data/hoper_sequence_representations/modal_rep_ae_node2vec_binary_fused_representations_dataframe_multi_col.csv
  launcher "$(config sae "    choice_of_module: [SimpleAe]
    representation_path: $fused
    simple_ae_module: {output_dir: $out, epochs: 2, batch_size: 128}")" || return 1
  inenv HoloProtRep-AE python multimodal_representations/simple_ae.py inference \
    --fused_rep_path "$fused" \
    --model_load_path "$out/simple_ae_weights.pth" --scaler_load_path "$out/simple_ae_scaler.pkl" \
    --output_csv "$out/simple_ae_inference.csv" || return 1
  pycheck "
import pandas as pd
for f in ['simple_ae_representation.csv', 'simple_ae_inference.csv']:
    d = pd.read_csv('$out/' + f, nrows=3); print(f, d.shape); assert d.columns[0] == 'Entry' and d.shape[1] == 513"
}

# Real example data for the multimodal autoencoders: ProtT5 sequence vectors (1024-d), Node2vec PPI vectors (50-d)
# and the TF-IDF SVD-256 text vectors of the 300 proteins written by t_text_tfidf.
AE_SEQ=./data/hoper_sequence_representations/T5_UNIPROT_HUMAN.csv
AE_PPI=./data/hoper_case_study_example_data/representation_files/node2vec_d_50_p_0.5_q_0.25_multi_col.csv
AE_TEXT=./text_representations/representation_generation/tfidf_representations/uniprotpubmed_tfidf_vectors_svd256.csv

# expected number of proteins = proteins present in every given file
ae_expected_rows() {
  inenv HoloProtRep-AE python -c "
import sys, pandas as pd
sets = [set(pd.read_csv(p, usecols=['Entry'])['Entry']) for p in sys.argv[1:]]
print(len(set.intersection(*sets)))" "$@" | tail -1
}
check_rep() { # <csv> <expected rows> <expected columns incl. Entry>
  pycheck "
import numpy as np, pandas as pd
d = pd.read_csv('$1'); print('$1'.split('/')[-1], d.shape)
assert d.shape == ($2, $3), d.shape
assert d.columns[0] == 'Entry' and np.isfinite(d.iloc[:, 1:].values).all()"
}

t_multimodal_ae() {
  [ -f "$AE_TEXT" ] || { echo "$AE_TEXT missing: run the t_text_tfidf step first"; return 1; }
  local out="$TMP/multimodal_ae" n
  launcher "$(config mmae "    choice_of_module: [MultiModalAe]
    multimodal_ae_module: {seq_csv: $AE_SEQ, ppi_csv: $AE_PPI, text_csv: $AE_TEXT, output_dir: $out, representation_dim: 512, epochs: 3, batch_size: 64}")" || return 1
  n=$(ae_expected_rows "$AE_SEQ" "$AE_PPI" "$AE_TEXT"); echo "proteins in all three inputs: $n"
  [ "$n" -gt 0 ] || return 1
  check_rep "$out/multimodal_ae_representation.csv" "$n" 513 && ls "$out/multimodal_ae_weights.pth" "$out/multimodal_ae_loss.png"
}

test_seq_input() { # sequence-only proteins for test mode: the t_sequence output, else 10 ProtT5 vectors
  if [ -f "$TMP/prott5_bfd.csv" ]; then echo "$TMP/prott5_bfd.csv"
  else head -n 11 "$AE_SEQ" > "$TMP/test_seq.csv"; echo "$TMP/test_seq.csv"; fi
}

t_transfer_ae() {
  local mm="$TMP/multimodal_ae/multimodal_ae_weights.pth"
  [ -f "$mm" ] || { echo "$mm missing: run the t_multimodal_ae step first"; return 1; }
  local out="$TMP/transfer_ae" test_seq n
  test_seq=$(test_seq_input)
  launcher "$(config trae "    choice_of_module: [TransferAe]
    transfer_ae_module: {seq_csv: $AE_SEQ, ppi_csv: $AE_PPI, text_csv: $AE_TEXT, multimodal_weights: $mm, test_seq_csv: $test_seq, output_dir: $out, epochs: 3, batch_size: 64}")" || return 1
  n=$(ae_expected_rows "$AE_SEQ" "$AE_PPI" "$AE_TEXT"); echo "proteins in all three inputs: $n"
  check_rep "$out/transfer_ae_representation.csv" "$n" 513 || return 1
  check_rep "$out/transfer_ae_test_representation.csv" "$(ae_expected_rows "$test_seq")" 513 || return 1
  ls "$out/transfer_ae_weights.pth.shift_factors.txt" "$out/transfer_ae_weights.pth.scaling_factors.txt"
}

t_transfer_ae_seq_text() {
  [ -f "$AE_TEXT" ] || { echo "$AE_TEXT missing: run the t_text_tfidf step first"; return 1; }
  local out="$TMP/transfer_ae_seq_text" test_seq n
  test_seq=$(test_seq_input)
  launcher "$(config trst "    choice_of_module: [TransferAeSeqText]
    transfer_ae_seq_text_module: {seq_csv: $AE_SEQ, text_csv: $AE_TEXT, test_seq_csv: $test_seq, output_dir: $out, representation_dim: 512, dual_epochs: 3, transfer_epochs: 3, batch_size: 64}")" || return 1
  n=$(ae_expected_rows "$AE_SEQ" "$AE_TEXT"); echo "proteins in both inputs: $n"
  check_rep "$out/dual_ae_representation.csv" "$n" 513 || return 1
  check_rep "$out/transfer_ae_seq_text_representation.csv" "$n" 513 || return 1
  check_rep "$out/transfer_ae_seq_text_test_representation.csv" "$(ae_expected_rows "$test_seq")" 513
}

t_case_study() {
  rm -rf case_study/case_study_results
  inenv hoper_case_study_env python case_study_main.py || return 1
  local pred=case_study/case_study_results/prediction/modal_rep_ae_prediction_binary_classifier_Fully_Connected_Neural_Network.csv
  ls case_study/case_study_results/training/modal_rep_ae_Fully_Connected_Neural_Network_binary_classifier.pt || return 1
  pycheck "
import pandas as pd; d = pd.read_csv('$pred'); print(d.shape, d['Label'].value_counts().to_dict())
assert d.shape == (1085, 2) and set(d['Label']) <= {0, 1}"
}

step "PPI: Node2vec + HOPE (launcher)"          t_ppi
[ $QUICK -eq 0 ] && step "sequence: ProtT5-XL bfd (launcher)" t_sequence
step "fuse_representations (launcher)"         t_fuse
step "text: TF-IDF, 300 proteins (launcher)"   t_text_tfidf
[ $QUICK -eq 0 ] && step "text: BioBERT, 20 proteins (launcher)" t_text_biobert
step "text: BioSentVec fails clearly w/o model" t_text_missing_model
step "text: visualization (launcher)"          t_visualize
[ $QUICK -eq 0 ] && step "Preprocessing, 300 entries (launcher)" t_preprocess
step "SimpleAE train (launcher) + inference"   t_simple_ae
step "MultiModalAE, real data (launcher)"      t_multimodal_ae
step "TransferAE seq+PPI+text, real data (launcher)" t_transfer_ae
step "TransferAE seq+text, real data (launcher)" t_transfer_ae_seq_text
step "case_study_main.py (example config)"     t_case_study

echo; echo "======== Summary"
failed=0
for i in "${!NAMES[@]}"; do
  printf "  %-42s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}"
  [[ "${RESULTS[$i]}" == PASS* ]] || failed=1
done
exit $failed
