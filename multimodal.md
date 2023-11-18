# SimpleAE example
1.For the models to work,intact.zip and uniprot_sprot.xml.gz files for uniprot preprocessing must be downloaded from the links below. 
- [Data files](https://drive.google.com/file/d/1R7jRfnBWmO6i6S1vqQd6zZt2-kcK6Eom/view?usp=drive_link)
- [Uniprot preprocessing data](https://drive.google.com/file/d/1fOu7cWX9f-B-Ro41VvLGgG8eyGhV8IwD/view?usp=drive_link)
-  [IntAct database](https://drive.google.com/file/d/1dblRYA3A-MH08iJDJm7L8MBoDMPOiS_g/view?usp=drive_link)
2. unzip data.zip under **Hoper** directory:
   '''
   unzip HOPER/data.zip
   '''
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
