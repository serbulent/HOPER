# SimpleAE example

1. Install HOPER and download the example data (see [README.md](README.md#installation)):

    ```shell
    bash create_env.sh
    bash download_data.sh
    ```

1. In `Hoper_representation_generetor.yaml` set `choice_of_module` to `SimpleAe` and point `representation_path`
   to a multi-column representation file (`Entry`, `0`, `1`, ...):

    ```yaml
    choice_of_module: [SimpleAe]

    #*******************SimpleAe*********************************************
    module_name: SimpleAe
    representation_path: ./data/hoper_sequence_representations/modal_rep_ae_node2vec_binary_fused_representations_dataframe_multi_col.csv
    simple_ae_module:
        output_dir: ./outputs
        epochs: 400
    ```

1. Run the simple autoencoder (about 30 minutes on CPU for 400 epochs):

    ```shell
    conda activate hoper
    python Hoper_representation_generetor_main.py
    ```

   Outputs: `outputs/simple_ae_representation.csv`, `outputs/simple_ae_weights.pth`, `outputs/simple_ae_scaler.pkl`,
   `outputs/simple_ae_loss.png`.

MultiModalAE and TransferAE are run the same way (`choice_of_module: [MultiModalAe]` / `[TransferAe]`); see
[multimodal_representations/readme.md](multimodal_representations/readme.md).
