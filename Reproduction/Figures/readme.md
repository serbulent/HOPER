# TSNE (T-distributed Stochastic Neighbor Embedding)
## Parameters

Used default parameters except perplexity (10) (perplexity :float, default=30.0)

! The perplexity must be less than the number of samples.

## Dependencies

1.Pandas 

2.Numpy

3.Seaborn

4.Random

5.Matplotlib

6.StandardScaler

7.bioinfokit

8.cluster

9.TSNE

10.sklearn

11.decomposition

### Data Required For the Function

* go_category_dataframe :Data containing GO ids

* multi_col_representation_processdata

* aspect,num_cat,termsif

example: 

           aspect=['MF']

           num_cat=['Low']
           
           termsif=['Shallow']

Example of TSNE file see tsne_def.py
