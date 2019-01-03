# Semnet: tools for metascience
Semnet is a series of modules for working with semantic networks. More specifically, we implemented algorithms that would help us gather features and rank nodes in terms of their connections to other nodes. The software was designed to work with a [network](https://skr3.nlm.nih.gov/SemMedDB/index.html) of biomedical concepts stored in a locally hosted [Neo4j](https://neo4j.com/) database ([node types](https://www.nlm.nih.gov/research/umls/META3_current_semantic_types.html), [edge types](https://www.nlm.nih.gov/research/umls/META3_current_relations.html)).

## Features
The following features have been implemented in Semnet:
- **Utilities**
	- Neo4j database interaction
	- Query parallelization
	- Vectorization
	- Negative example generation
- **Feature Extraction**
	- Metapath counts between node pairs
	- Degree-weighted path counts between node pairs
	- HeteSim similarity scores between node pairs
- **Computation**
	- Latent semantic relationship extraction
	- User-guided similarity search
	- Unsupervised ranker aggregation

## Citations
> Himmelstein, Daniel S., and Sergio E. Baranzini. "Heterogeneous network edge prediction: a data integration approach to prioritize disease-associated genes." PLoS computational biology 11.7 (2015): e1004259.

> Klementiev, Alexandre, Dan Roth, and Kevin Small. "An unsupervised learning algorithm for rank aggregation." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2007.

> Shi, Chuan, et al. "HeteSim: A General Framework for Relevance Measure in Heterogeneous Networks." IEEE Trans. Knowl. Data Eng. 6.10 (2014): 2479-2492.

> Wang, Chenguang, et al. "Relsim: relation similarity search in schema-rich heterogeneous information networks." Proceedings of the 2016 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2016.

> Yu, Xiao, et al. "User guided entity similarity search using meta-path selection in heterogeneous information networks." Proceedings of the 21st ACM international conference on Information and knowledge management. Acm, 2012.