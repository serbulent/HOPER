# t-SNE (t-Distributed Stochastic Neighbor Embedding)
t-SNE  is a machine learning algorithm commonly used for visualizing high-dimensional data in a lower-dimensional space.

The primary goal of t-SNE is to represent each data point in a lower-dimensional space, typically two or three dimensions, while preserving the pairwise similarities between data points as much as possible. It is particularly useful for visualizing complex datasets that cannot be easily visualized in their original high-dimensional space. 
## Parameters

Used default parameters except perplexity (10) (perplexity :float, default=30.0)

- The perplexity must be less than the number of samples.

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

* multi_col_representation_processdata : Example of processdata file see tsne_preprocess_data.py

* aspect,num_cat,termsif

example: 

           aspect=['MF']

           num_cat=['Low']
           
           termsif=['Shallow']

Example of TSNE file see tsne_def.py

### RUN TSNE

* GO_IDs are selected as labels in the data.
* The part other than the label is selected as train and the model is run.

