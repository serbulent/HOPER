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


if __name__ == "__main__":
    main()
