Rank Aggregation
================

After we have computed vectorized representations of the connections between source-target pairs using the :ref:`FeatExtract` modules, it can be difficult to interpret these feature vectors. We have found it useful to run a ranking aggregation algorithm on these features [#]_. This treats each metapath as an independent ranker that ranks each source node with respect to each target node. 

One method of aggregating this information th


.. [#] Klementiev, Alexandre, Dan Roth, and Kevin Small. "An unsupervised learning algorithm for rank aggregation." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2007.