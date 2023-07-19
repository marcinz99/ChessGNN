# ChessGNN

This repository accompanies my Master Thesis in Computer Science, “Exploring the applicability of graph neural networks for chess engines”. Graph neural networks (Graph Networks specifically) are proposed there to be used for the analysis of the board states for the move prediction (in particular direct move prediction).

Please, keep in mind that the work was primarily focused on the GNNs themselves. In particular, it is shown how convenient it is to perform graph-based tasks for different tasks that can be defined.

This repo is meant mostly for the archival purposes.

### Abstract of the Thesis

> This work introduces a novel idea of using the graph neural networks (GNNs) for deep learning-based chess engines. Its main focus is on assessing their applicability and impact in such a, yet unexplored, scenario. The usage of the graph neural networks here is supported by the notion of a more straight-forward interpretation of the graph structure of the pieces on the board and their mutual relationships, therefore potentially allowing it to be more efficiently exploited this way. Thus, deep learning strategies, and graph neural networks in particular, are used here to develop a (hopefully) potent chess engine, requiring as little human expertise as possible, except for basic rules. The scope of this work includes defining a proper graph representation for chess games and information propagation through the layers of the graph network, as well as proposing an appropriate training schema. Notably, the findings of this work can be applied to other problems than chess, as the general idea is agnostic of such details. Undeniable advantage of graph representation is its conciseness both for inputs and outputs. Unfortunately, however, the results obtained in this work are not truly decisive, as no evidence of the superiority of GNNs have been shown at this point – it does, nonetheless, provide a rather satisfactory and elaborate proof of concept for further research efforts.

### Repo contents

1. `chessgnn` is a tiny custom-made library implementing the GNN operations used for the implementation of the deep learning model.
2. `notebooks` contains the `.ipynb` files with final Google Colab notebooks used in the process. These are provided for the additional insight into all parts of the project (e.g. see notebook 3A for the definition of the model). Running them might require some slight modifications.
3. `weights` contains the final `.h5` files of the trained models at two stages of the training (v1 is after 100 training episodes (CCRL data), v2 is after additional 30 episodes (human data)), just as stated in the "Experiments" section of the thesis (you may consider checking it for more information). To load the model please use `load_model` function provided in the `chessgnn` library.

### More info about the custom GNN code

* Based on TensorFlow 2 (Keras) - compatible with the layer-oriented approach, and thus fully supporting the functional API of Keras.
* Subscribes to the idea of the conceptual framework known as Graph Networks, although implements it by specifying low-level building blocks, providing quite a lot of flexibility:
  * Scatter and aggregate – transformation from edge domain to node domain; aggregates (sum, mean or max) edge features according to the target nodes;
  * Scatter and softmax – applies softmax in a segment-wise manner in the edge domain according to the target nodes;
  * Concat edges’ ends only – transformation from node domain to edge domain; concatenates features of both ends of the edges;
  * Concat edges with single end or both ends – enriches edge level features with appropriate node features;
  * Concat with broadcast (concat globals with anything) – appends global features onto the edge or node level.
* Assumes the batch size of 1.
