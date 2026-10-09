"""HOPER representation generator.

Usage (from the repository root, in the ``hoper`` environment):

    python Hoper_representation_generetor_main.py [config.yaml]

Reads ``Hoper_representation_generetor.yaml`` (or the given file) and runs
each selected module inside its own conda environment (via ``conda run``).

This launcher itself only needs ``pyyaml``; module dependencies live in the
module environments created by ``create_env.sh``.
"""
import json
import os
import shutil
import subprocess
import sys

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required to run HOPER: pip install pyyaml")

HOPER_BASE = os.path.dirname(os.path.abspath(__file__))

ENV_PPI = "hoper_PPI"
ENV_TEXT = "HOPER_textrepresentations"
ENV_PREPROCESS = "hoper_preprocess"
ENV_CASE_STUDY = "hoper_case_study_env"
ENV_AE = "HoloProtRep-AE"
ENV_SEQUENCE = "prott5xl"


def find_conda():
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        sys.exit("conda was not found. Activate a conda installation before running HOPER.")
    return conda


def run_in_env(env, args, extra_env=None):
    """Run ``python <args>`` in a conda environment and stop on failure."""
    cmd = [find_conda(), "run", "--no-capture-output", "-n", env, "python"] + [str(a) for a in args]
    print("\n>>> [{}] {}".format(env, " ".join(cmd[6:])), flush=True)
    env_vars = dict(os.environ)
    env_vars["PYTHONPATH"] = HOPER_BASE + os.pathsep + env_vars.get("PYTHONPATH", "")
    if extra_env:
        env_vars.update(extra_env)
    result = subprocess.run(cmd, cwd=os.getcwd(), env=env_vars)
    if result.returncode != 0:
        sys.exit("HOPER: step failed in environment '{}' (exit code {}).".format(env, result.returncode))


INPUT_HINTS = {
    "tfidf_representations": "run the text module with tfidf first (choice_of_module: [text, ...])",
    "prott5_": "run the sequence module first, or set test_seq_csv to an existing file or ''",
    "multimodal_ae_weights": "run the MultiModalAe module first (choice_of_module: [..., MultiModalAe, TransferAe])",
}


def require_inputs(module, *paths):
    """Stop before a long training run if an input file is missing."""
    for path in paths:
        if not os.path.isfile(path):
            hint = next((h for key, h in INPUT_HINTS.items() if key in path), "check the path in the config file")
            sys.exit("HOPER {}: input file not found: {}\n  -> {}".format(module, path, hint))


def first(value):
    """YAML values are written as one-element lists in the examples; accept both forms."""
    return value[0] if isinstance(value, list) else value


def main():
    # Optional argument: path of the config file (default: ./Hoper_representation_generetor.yaml)
    config_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "Hoper_representation_generetor.yaml")
    os.environ["HOPER_CONFIG"] = config_path  # read by the preprocessing modules
    with open(config_path, "r") as stream:
        params = yaml.safe_load(stream)["parameters"]
    modules = params["choice_of_module"]
    is_directed = bool(params.get("is_directed", False))

    if "PPI" in modules:
        edge_f = first(params["interaction_data_path"])
        protein_id = first(params["protein_id_list"])
        reps = params["choice_of_representation_name"]
        if "Node2vec" in reps:
            n2v = params["node2vec_module"]["parameter_selection"]
            run_in_env(ENV_PPI, ["ppi_representations/Node2vec.py", edge_f, protein_id, is_directed,
                                 json.dumps(n2v["d"]), json.dumps(n2v["p"]), json.dumps(n2v["q"])])
        if "HOPE" in reps:
            hope = params["HOPE_module"]["parameter_selection"]
            run_in_env(ENV_PPI, ["ppi_representations/HOPE.py", edge_f, protein_id, is_directed,
                                 json.dumps(hope["d"]), json.dumps(hope["beta"])])

    if "sequence" in modules:
        seq = params["sequence_module"]
        run_in_env(ENV_SEQUENCE, ["sequence_representations/prott5xl.py",
                                  "--input", seq["input_path"], "--output", seq["output_path"],
                                  "--model", seq.get("model", "bfd"),
                                  "--batch_size", seq.get("batch_size", 8)])

    if "Preprocessing" in modules:
        run_in_env(ENV_PREPROCESS, ["text_representations/preprocess/preprocess_main.py"])

    if "text" in modules:
        processes = params["choice_of_process"]
        if "generate" in processes:
            gen = params["generate_module"]
            args = ["text_representations/representation_generation/createtextrep.py"]
            args += ["--" + rep for rep in gen["choice_of_representation_type"]]
            args += ["-upfp", first(gen["uniprot_files_path"]), "-pmfp", first(gen["pubmed_files_path"]),
                     "-mdw", gen.get("model_download", "n")]
            run_in_env(ENV_TEXT, args)
        if "visualize" in processes:
            vis = params["visualize_module"]
            run_in_env(ENV_TEXT, ["text_representations/result_visualization/visualize_results.py",
                                  "-" + first(vis["choice_of_visualization_type"]),
                                  "-rfp", first(vis["result_files_path"])])

    if "fuse_representations" in modules:
        run_in_env(ENV_CASE_STUDY, ["utils/fuse_representation.py",
                                    json.dumps(params["representation_files"]),
                                    str(params["min_fold_number"]),
                                    json.dumps(params["representation_names"])])

    if "SimpleAe" in modules:
        ae = params.get("simple_ae_module", {})
        out_dir = ae.get("output_dir", "./outputs")
        run_in_env(ENV_AE, ["multimodal_representations/simple_ae.py", "train",
                            "--fused_rep_path", params["representation_path"],
                            "--model_save_path", os.path.join(out_dir, "simple_ae_weights.pth"),
                            "--scaler_save_path", os.path.join(out_dir, "simple_ae_scaler.pkl"),
                            "--output_csv", os.path.join(out_dir, "simple_ae_representation.csv"),
                            "--loss_plot_path", os.path.join(out_dir, "simple_ae_loss.png"),
                            "--epochs", ae.get("epochs", 400),
                            "--batch_size", ae.get("batch_size", 128),
                            "--learning_rate", ae.get("learning_rate", 0.001),
                            "--validation_split", ae.get("validation_split", 0.2),
                            "--seed", ae.get("seed", 42)])

    if "MultiModalAe" in modules:
        mm = params["multimodal_ae_module"]
        out_dir = mm.get("output_dir", "./outputs")
        require_inputs("MultiModalAe", mm["seq_csv"], mm["ppi_csv"], mm["text_csv"])
        run_in_env(ENV_AE, ["multimodal_representations/multi_odal_representations.py",
                            "--seq_csv", mm["seq_csv"], "--ppi_csv", mm["ppi_csv"], "--text_csv", mm["text_csv"],
                            "--representation_dim", mm.get("representation_dim", 512),
                            "--epochs", mm.get("epochs", 100),
                            "--batch_size", mm.get("batch_size", 128),
                            "--lr", mm.get("learning_rate", 0.001),
                            "--seed", mm.get("seed", 42),
                            "--save_model_path", os.path.join(out_dir, "multimodal_ae_weights.pth"),
                            "--save_csv_path", os.path.join(out_dir, "multimodal_ae_representation.csv"),
                            "--loss_plot_path", os.path.join(out_dir, "multimodal_ae_loss.png")])

    if "TransferAe" in modules:
        # sequence-only model initialised from the MultiModalAE (sequence + PPI + text)
        tr = params["transfer_ae_module"]
        out_dir = tr.get("output_dir", "./outputs")
        transfer_weights = os.path.join(out_dir, "transfer_ae_weights.pth")
        require_inputs("TransferAe", tr["seq_csv"], tr["ppi_csv"], tr["text_csv"], tr["multimodal_weights"],
                       *([tr["test_seq_csv"]] if tr.get("test_seq_csv") else []))
        run_in_env(ENV_AE, ["multimodal_representations/transfer_ae.py", "--mode", "train",
                            "--seq_csv", tr["seq_csv"], "--ppi_csv", tr["ppi_csv"], "--text_csv", tr["text_csv"],
                            "--model_weights", tr["multimodal_weights"],
                            "--epochs", tr.get("epochs", 200),
                            "--batch_size", tr.get("batch_size", 128),
                            "--seed", tr.get("seed", 42),
                            "--save_model_path", transfer_weights,
                            "--save_csv_path", os.path.join(out_dir, "transfer_ae_representation.csv"),
                            "--loss_plot_path", os.path.join(out_dir, "transfer_ae_loss.png")])
        if tr.get("test_seq_csv"):
            run_in_env(ENV_AE, ["multimodal_representations/transfer_ae.py", "--mode", "test",
                                "--seq_csv", tr["test_seq_csv"], "--model_weights", transfer_weights,
                                "--save_csv_path", os.path.join(out_dir, "transfer_ae_test_representation.csv")])

    if "TransferAeSeqText" in modules:
        # sequence + text variant: dual autoencoder, then a sequence-only model initialised from it
        tr = params["transfer_ae_seq_text_module"]
        out_dir = tr.get("output_dir", "./outputs")
        dual_weights = os.path.join(out_dir, "dual_ae_weights.pth")
        transfer_weights = os.path.join(out_dir, "transfer_ae_seq_text_weights.pth")
        require_inputs("TransferAeSeqText", tr["seq_csv"], tr["text_csv"],
                       *([tr["test_seq_csv"]] if tr.get("test_seq_csv") else []))
        # 1) sequence + text autoencoder, 2) sequence-only TransferAE initialised from it
        run_in_env(ENV_AE, ["multimodal_representations/multimodal_text_seq.py",
                            "--seq_csv", tr["seq_csv"], "--text_csv", tr["text_csv"],
                            "--representation_dim", tr.get("representation_dim", 512),
                            "--epochs", tr.get("dual_epochs", 100),
                            "--batch_size", tr.get("batch_size", 128),
                            "--seed", tr.get("seed", 42),
                            "--save_model_path", dual_weights,
                            "--save_csv_path", os.path.join(out_dir, "dual_ae_representation.csv"),
                            "--loss_plot_path", os.path.join(out_dir, "dual_ae_loss.png")])
        run_in_env(ENV_AE, ["multimodal_representations/transfer_text_seq.py", "--mode", "train",
                            "--seq_csv", tr["seq_csv"], "--text_csv", tr["text_csv"],
                            "--model_weights", dual_weights,
                            "--epochs", tr.get("transfer_epochs", 200),
                            "--batch_size", tr.get("batch_size", 128),
                            "--seed", tr.get("seed", 42),
                            "--save_model_path", transfer_weights,
                            "--save_csv_path", os.path.join(out_dir, "transfer_ae_seq_text_representation.csv"),
                            "--loss_plot_path", os.path.join(out_dir, "transfer_ae_seq_text_loss.png")])
        # 3) optional: proteins that only have a sequence representation
        if tr.get("test_seq_csv"):
            run_in_env(ENV_AE, ["multimodal_representations/transfer_text_seq.py", "--mode", "test",
                                "--seq_csv", tr["test_seq_csv"], "--model_weights", transfer_weights,
                                "--save_csv_path", os.path.join(out_dir, "transfer_ae_seq_text_test_representation.csv")])


if __name__ == "__main__":
    main()
