#!/usr/bin/env bash
# Creates (or updates) every HOPER conda environment, installs GEM and builds the SNAP node2vec binary.
# Each step runs independently; a summary is printed at the end and the exit code is non-zero if any step failed.
# Requirements: conda (Miniconda/Miniforge/Anaconda) and git. No system compiler is needed.
set -u
cd "$(dirname "$0")"
HOPER_BASE=$(pwd)

GEM_COMMIT=213189b
SENT2VEC_COMMIT=9efbc2dd69f6c737c3a752c9dc5fbb4843d578b6
SNAP_COMMIT=6924a035aabd1ce0a547b94e995e142f29eb5040

command -v conda >/dev/null 2>&1 || { echo "conda not found: install Miniconda/Miniforge first."; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git not found: it is needed to install GEM, sent2vec and SNAP."; exit 1; }

STEPS=()
STATUS=()

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

make_env() { # <env name> <yml>
  if env_exists "$1"; then
    conda env update -n "$1" -f "$2"
  else
    conda env create -n "$1" -f "$2"
  fi
}

run_step() { # <label> <command...>
  local label=$1; shift
  echo; echo "======== $label"
  if "$@"; then STATUS+=("OK"); else STATUS+=("FAILED"); fi
  STEPS+=("$label")
}

build_node2vec() {
  local snap_dir="$HOPER_BASE/ppi_representations/snap"
  env_exists hoper_build || conda create -y -n hoper_build -c conda-forge make cxx-compiler || return 1
  if [ ! -d "$snap_dir/.git" ]; then
    git clone https://github.com/snap-stanford/snap "$snap_dir" || return 1
  fi
  git -C "$snap_dir" checkout -q "$SNAP_COMMIT" || return 1
  conda run --no-capture-output -n hoper_build bash -c \
    "make -C '$snap_dir/examples/node2vec' CC=\"\${CXX:-g++}\"" || return 1
  local bin_dir="$HOPER_BASE/ppi_representations/bin"
  local build_prefix
  build_prefix=$(conda run -n hoper_build python -c 'import sys; print(sys.prefix)' 2>/dev/null || conda run -n hoper_build bash -c 'echo $CONDA_PREFIX')
  mkdir -p "$bin_dir"
  cp "$snap_dir/examples/node2vec/node2vec" "$bin_dir/node2vec"
  # Ship the C++/OpenMP runtimes next to the binary; Node2vec.py adds this directory to LD_LIBRARY_PATH.
  for lib in libgomp.so.1 libstdc++.so.6 libgcc_s.so.1; do
    cp -L "$build_prefix/lib/$lib" "$bin_dir/" || return 1
  done
  LD_LIBRARY_PATH="$bin_dir" "$bin_dir/node2vec" >/dev/null 2>&1
  [ $? -ne 127 ]
}

run_step "hoper (launcher)"            make_env hoper environment.yml
run_step "hoper_case_study_env"        make_env hoper_case_study_env case_study/hoper_case_study_env.yml
run_step "hoper_PPI"                   make_env hoper_PPI ppi_representations/hoper_PPI.yml
run_step "GEM @$GEM_COMMIT"            conda run -n hoper_PPI pip install --no-deps "git+https://github.com/palash1992/GEM.git@$GEM_COMMIT"
run_step "node2vec (SNAP)"             build_node2vec
run_step "hoper_preprocess"            make_env hoper_preprocess text_representations/preprocess/hoper_preprocess.yml
run_step "HOPER_textrepresentations"   make_env HOPER_textrepresentations text_representations/text_representations.yml
run_step "sent2vec (epfml)"            conda run --no-capture-output -n HOPER_textrepresentations pip install --no-build-isolation "git+https://github.com/epfml/sent2vec.git@$SENT2VEC_COMMIT"
run_step "HoloProtRep-AE"              make_env HoloProtRep-AE multimodal_representations/simple_ae_env.yml

echo; echo "======== Summary"
failed=0
for i in "${!STEPS[@]}"; do
  printf "  %-30s %s\n" "${STEPS[$i]}" "${STATUS[$i]}"
  [ "${STATUS[$i]}" = "OK" ] || failed=1
done
exit $failed
