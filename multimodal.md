# SimpleAE example

1. create the necessary environment with:

      python create_env.py 

2. Make manipulations on Hoper_representation_generetor.yaml file for generating multimodal representations. Change the "choice_of_module" parameter to SimpleAe and provide the representation_path parameter as your representation path.

```
choice_of_module: [SimpleAe] # Module selection 

#*******************SimpleAe*********************************************
    module_name: SimpleAe
    representation_path: ./data/hoper_sequence_representations/modal_rep_ae_node2vec_binary_fused_representations_dataframe_multi_col.csv
```
3. Run Simple autoencoder for multimodal representations
```	
	python Hoper_representation_generetor_main.py
```
