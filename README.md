# Efficient Sampling Techniques for Embedding Large Graphs
- Kijin Kwon, Juyeong Shin, Young-Koo Lee
- KCC 2022 Paper
- Paper link: [https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11113618](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11113618)<br/><br/>
- Modified from DGL pinSAGE example
   - paper link: [https://dl.acm.org/doi/abs/10.1145/3219819.3219890](https://dl.acm.org/doi/abs/10.1145/3219819.3219890)
   - example link: [https://github.com/dmlc/dgl/tree/master/examples](https://github.com/dmlc/dgl/tree/master/examples)

## Requirements
- dask
- pandas
- pytorch 2.0.1
- dgl 1.1.1
- torchtext >= 0.9.0

## Prepare datasets

### MovieLens 1M

1. Download and extract the MovieLens-1M dataset from http://files.grouplens.org/datasets/movielens/ml-1m.zip
   into the current directory.
2. Run `python process_movielens1m.py ./ml-1m ./data_processed`.
   Replace `ml-1m` with the directory you put the `.dat` files, and replace `data_processed` with
   any path you wish to put the output files.

### Nowplaying-rs

1. Download and extract the Nowplaying-rs dataset from https://zenodo.org/record/3248543/files/nowplayingrs.zip?download=1
   into the current directory.
2. Run `python process_nowplaying_rs.py ./nowplaying_rs_dataset ./data_processed`

## Run model

### Nearest-neighbor recommendation

This model returns items that are K nearest neighbors of the latest item the user has
interacted.  The distance between two items are measured by Euclidean distance of
item embeddings, which are learned as outputs of PinSAGE.

```
python model.py data_processed --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 64
```

The implementation here also assigns a learnable vector to each item.  If your hidden
state size is so large that the learnable vectors cannot fit into GPU, use this script
for sparse embedding update (written with `torch.optim.SparseAdam`) instead:


```
python model_sparse.py data_processed --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 1024
```

Note that since the embedding update is done on CPU, it will be significantly slower than doing
everything on GPU.

The HITS@10 is 0.01241, compared to 0.01220 with SLIM with the same dimensionality.

## Difference from the pinSAGE paper with DGL example

The implementation here is different from what being described in the paper:

1. The paper described a supervised setting where the authors have a ground truth set of which items are
   relevant.  However, in traditional recommender system datasets we don't have such labels other than
   which items are interacted by which users (as well as the user/item's own features).  Therefore, I
   adapted PinSAGE to an unsupervised setting where I predict whether two items are cointeracted by the
   same user.
2. PinSAGE paper explicitly stated that the items do not learnable embeddings of nodes, but directly
   express the embeddings as a function of node features.  While this is reasonable for rich datasets like
   Pinterest's where images and texts are rich enough to distinguish the items from each other, it is
   unfortunately not the case for traditional recommender system datasets like MovieLens or Nowplaying-RS
   where we only have a bunch of categorical or numeric variables.  I found adding a learnable embedding
   for each item still helpful for those datasets.
3. The PinSAGE paper directly pass the GNN output to an MLP and make the result the final item
   representation.  Here, I'm adding the GNN output with the node's own learnable embedding as
   the final item representation instead.

### LICENSE
© 2022 meongju0o0 uses Apache 2.0 License. Powered by DGL Team.
