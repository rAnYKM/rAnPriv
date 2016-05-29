# De-anonymizing Social Networks and Inferring Private Attributes Using Knowledge Graphs

## Introduction

**Limitations of Previous Works**

1. Focusing on de-anonymization (identification), but barely privacy inference after de-anonymization
2. Having specific assumptions about the attacker's prior knowledge (background information)
3. Typically neglecting the correlations among attributes tp make inference about users' sensitive attributes

**Goal**

To construct a comprehensive and realistic model of the attacker's knowledge and use this model to depict the privacy inferring process

**Challenges**

1. Various background knowledge, varying from node profiles and degrees to link relations
2. Modeling the privacy inference steps
3. Quantifying privacy disclosure

**Contributions**

1. Applying knowledge graph to model the attaker's background knowledge
2. Utilizing the model to depict the privacy inference process
3. Presenting an experiment of attack on two real life social network datasets (SNAP datasets)

## Modeling

**Knowledge Graph**

A network of all kinds of entities related to a specific domain or topic.

RDF tuple: 
````
	<subject, predicate/relation, object>
````

Each tuple is assigned a confidence score implying the probability of this knowledge being true.

**Social Network Data Model**

In this paper, a social network is modeled with a knowledge graph *G(V,E)* where *V* is a set of nodes representing entities of the network, *E* is a set of links representing the relations between them.

**Attack Model**

- *Publisher* : have full access to all the information
- *Attacker* : aim to learn private inforamtion of a specific target with access to published datasets and a variety of background information (prior knowledge)

**Attacker's Knowledge Model**

Prior Knowledge Types

- Common Sense
- Statistical Information
- Personal Information
- Network Structural Information

Knowledge Correlation Types

- Mutual Exclusion
- Inclusion
- Soft Correlation

## Knowledge Graph Based Attack

- Prior Attack Graph Construction

	- Initialize with certain knowledge
	- Add probabilistic knowledge
	- Complement according to correlations
	- Obtain published anonymous data graph G<sub>a</sub> and prior attack graph G<sub>p</sub>

- De-anonymization

	Map the targeted user in G<sub>p</sub> to G<sub>a</sub> based on the similarity of attributes and relations

- Privacy Inference

	Complement and update the attack graph by inferring private attributes and relations not in G<sub>a</sub>

- Privacy Disclosure Determination

	Define whether a knowledge triple *t* is considered to be disclosed by distance function

## Knowledge Graph Based Methods

- Node Similarity: Attribute & Structual
- De-anonymization Formulation: Construct Bipartite Graph 
- Path Ranking Based Privacy Inference

## Experiments

- Data Set: SNAP Google+ and Pokec

- Anonymized Graph Generation

- Evaluation



	

