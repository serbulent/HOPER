# SimpleAE example

1. Download the data files: 

   - [Data files](https://drive.google.com/file/d/1R7jRfnBWmO6i6S1vqQd6zZt2-kcK6Eom/view?usp=drive_link)

1. Place data.zip file under **Hoper** directory:

    ```shell
    cd HOPER
    unzip data.zip
    ```

1. create the necessary environment with:

    ```shell
    python create_env.py 
    ```

1. Make manipulations on Hoper_representation_generetor.yaml file for generating multimodal representations. Change the "choice_of_module" parameter to SimpleAe and provide the representation_path parameter as your representation path.

    ```yaml
    choice_of_module: [SimpleAe] # Module selection 

    #*******************SimpleAe*********************************************
        module_name: SimpleAe
        representation_path: ./data/hoper_sequence_representations/modal_rep_ae_node2vec_binary_fused_representations_dataframe_multi_col.csv
    ```

1. Run Simple autoencoder for multimodal representations

    ```shell	
        python Hoper_representation_generetor_main.py
    ```
