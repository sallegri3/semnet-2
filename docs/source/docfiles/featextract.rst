Feature Extraction
==================

Overview
--------

We can represent each pair of nodes in the biomedical concept graph in terms of the many types of paths between them. For example, there might be hundreds of genes and proteins between the amyloid-:math:`\beta` (:term:`source`) and Alzheimer's disease (:term:`target`) nodes. Additionally, the specific relationships that connect these genes and proteins with amyloid-:math:`\beta` and Alzheimer's disease are likely to vary. Each gene or protein could even have multiple relationships with both the source and target nodes. We use the term :term:`paths` to refer to the *specific sequences of nodes and relationships* that lead from the source node to the target node. More generally, we can use the term :term:`metapaths` to refer to the *sequences of node and relationship types* between source and target nodes.

.. note:: Since each node has both a unique identity and one of a set of 133 types, each metapath can represent multiple paths.

When characterizing the relationship between a pair of nodes in the graph, we first find the set of metapaths between the two nodes. Then, for each of these metapaths, we compute a feature value. The simplest example of a metapath-based feature is simply the total number of paths associated with that metapath. We have created classes and functions for extracting three types of metapath-based features from the graph.

.. image:: ../_static/feature_extraction.png

In addition to counts, we can compute two additional metrics - :term:`degree-weighted path count` (DWPC) and :term:`HeteSim`. Both of these features are designed to counteract the bias that highly connected nodes cause in the network. 

Usage
-----

Count Features
^^^^^^^^^^^^^^

The :class:`semnet.feature_extraction.CountExtractor` class is designed to find and count all metapaths between pairs of nodes. All you need to do is specify Python ``lists`` of the UMLS CUIs of source and target nodes. Notice that we repeat ``sources`` and ``targets`` in ``s`` and ``t`` to get all possible combinations.

.. code-block:: python

    import numpy as np
    from semnet.feature_extraction import CountExtractor

    cex = CountExtractor()
    s = sorted(sources) * len(targets)
    t = np.repeat(sorted(targets), len(sources))
    count_data = cex.get_all_metapath_counts(s, t, 2)

Extracting count-based features is usually fairly quick - only a few seconds per pair.

.. note:: When performing feature extraction, you will always start by collecting count-based features. This is because the process of collecting unique metapaths is integrated into this part of SemNet, and counting the paths doesn't add a substantial amount of time to the process.

DWPC Features
^^^^^^^^^^^^^

HeteSim Features
^^^^^^^^^^^^^^^^

Functions
---------

General Feature Extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: semnet.feature_extraction
    :members:

HeteSim
^^^^^^^

.. automodule:: semnet.hetesim
    :members:
