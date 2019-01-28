Feature Extraction
==================

Overview
--------

We can represent each pair of nodes in the biomedical concept graph in terms of the many types of paths between them. For example, there might be hundreds of genes and proteins between the Amyloid-Beta (**source**) and Alzheimer's disease (**target**) nodes. Additionally, the specific relationships that connect these genes and proteins with Amyloid-Beta and Alzheimer's disease are likely to vary. Each gene or protein could even have multiple relationships with both the source and target nodes. We use the term **paths** to refer to the *specific sequences of nodes and relationships* that lead from the source node to the target node. More generally, we can use the term **metapaths** to refer to the *sequences of node and relationship types* between source and target nodes.

.. note:: Since each node has both an identity and a type, each :term:`metapath` can represent multiple :term:`paths`.

When characterizing the relationship between a pair of nodes in the graph, we first find the set of metapaths between the two nodes. Then, for each of these metapaths, we compute a feature value. The simplest example of a metapath-based feature is simply the total number of paths associated with that metapath. We have created classes and functions for extracting three types of metapath-based features from the graph.

In addition to counts, we can compute two additional metrics - :term:`degree-weighted path count` (DWPC) and :term:`HeteSim`.

When we characterize We have created three for extracting three types of metapath-based features from the graph.


We call any specific sequence of relationships and nodes that connect the nodes of interest paths. SemNet feature extraction works by identifying all of the unique metapaths between
We use feature extraction modules to compute features on the graph. Currently, SemNet calculates metapath-based features.

.. image:: ../_static/feature_extraction.png

General Feature Extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: semnet.feature_extraction
    :members:

HeteSim
^^^^^^^

.. automodule:: semnet.hetesim
    :members:
