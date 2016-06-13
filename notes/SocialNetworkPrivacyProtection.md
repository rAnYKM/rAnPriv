
# Privacy-Preserving Online Social Network Publication

## Problem Desciption

### Motivation Senario

Nowadays, online social networks provide third party developers with APIs for application development. The third party applications can request user permissionw to access the information available the online social networks (user profile, user activities, friend circles, etc.) Although users can control their access permissions to avoid privacy disclosure, attackers still can infer private information (secrets) from public information. In addition, online social network users have different privacy concerns. Most existing work only considers to conceal secrets uniformly. To add this feature, we can use a matrix to record whether a user regard a certain attribute as a secret. 

### Problem Formulation

Given a complete online social attribute network $G=(V_N, V_A, E_N, E_A)$ and user privacy concern matrix $C \in \mathcal{N}^{m*n}$, obtian a new online social attribute network $G'=(V_N, V_A, E'_N, E'_A)$ so that user privacy concern can be satisfied with the least loss of utility.

where $V_N$ is the vertex set of social nodes(actors), $V_A$ is the vertex set of all attribute nodes (including private and public attributes), $E_N$ is the edge set of social relations and $E_A$ is the attribute link set between social nodes and attributes, $m=|V_A|, n=|V_N|$.

For a simplified version with only one specific attribute $A$, the user privacy concern is represented as a vector $C$ to show whether they want to conceal this attribute.


## Differential Privacy

Let $\epsilon$ be a positive real number and $\mathcal {A}$ be a randomized algorithm that takes a dataset as input (representing the actions of the trusted party holding the data). Let $\textrm {im} \mathcal {A}$ denote the image of $\mathcal {A}$ . The algorithm $\mathcal {A}$  is $\epsilon$ -differentially private if for all datasets $D_{1}$  and $D_{2}$  that differ on a single element (i.e., the data of one person), and all subsets $S$  of $\textrm {im} \mathcal {A}$ ,

$\Pr[{\mathcal {A}}(D_{1})\in S]\leq e^{\epsilon }\times \Pr[{\mathcal {A}}(D_{2})\in S],$ 
where the probability is taken over the randomness used by the algorithm.

## Common Neighbor Metrics

- **Jaccard Coefficient**: normalized common neighbors metric
$$J(x,y)=\frac{|\Gamma(x) \cap \Gamma(y)|}{|\Gamma(x) \cup \Gamma(y)|}$$

- **Adamic/Adar**: a metric of similarity between two social nodes (common neighbor case)
$$A(x,y)=\sum_{z \in \Gamma(x) \cap \Gamma(y)} {\frac{1}{\log{|\Gamma(z)|}}}$$

The common neighbor metrics can be utilized to evaluate the correlation among attributes in a social attribute network.

## Privacy Disclosure

(Qian‘s Solution) Given a piece of knowledge/triple $t$ which is considered to be sensitive to the user, $t$ has a score of $c_p(t)$ in $G_p$ and a score of $c_q(t)$ in $G_q$. The privacy $t$ is considered to be disclosed if 
$$\delta(c_p(t),c_q(t))>\epsilon (t)$$

(Li's Solution) An anonymiy graph $G^*$ is privacy preserving against attack capability defined by attack graph $G_A$ if all users satisfy
$$\delta(Pr_i,Pr'_t|G_A,G^*)\le \epsilon(id_i,s_j)$$

Both solutions focus on the correlation between public attributes and private attributes. In a partially labeled social network (a certain attribute is treated as public for some nodes and private for the others), attackers can use the labeled nodes and their relations to infer other nodes' sensitive information. Therefore, we need to find out the correlation between the sensitive attribute and social relations.


## Privacy Inference

### Without social relations (Li's Solution)

*Background Knowledge*
- **Prior probability of secrets**
- **Public attributes of nodes**

*Maximum Privacy Disclosure*
- No public attributes: the disclosure rate = prior probability $P(S)$
- All public attributes: the disclosure rate = $\max P(S|A)$

$$P(S|A) = \frac{P(A|S) \times P(S)}{P(A)}$$

### With social relations

*Treat as attributes* (Qian's solution)

Actually in Qian's attack model, the social relations are treated as the secrets. They use Path Ranking Algorithms to infer the private attributes and hidden links. And no more detailed explanation is provided.

Different from the correlation between public and private attributes, which can be obtained from statistics and common sense, the correlation between social relations and private attributes can hardly obtained from the perspective of attackers. However, in our case, since some of nodes' "secrets" (private only for the others) are exposed, the social relations can be utilized to infer. It is still feasible to find out the correlation through the exposed ones.

### Community Structure Destruction

We can notice that a group of people with similar properties will have dense network structure, which is the community. To defend the attack via community detection, the direct way is to conceal social relations to reduce the density so and destruct the community structure.


## Utility

### Statistical properties (Li's solution)

The Utility function is based on the difference between the original and anonymized graphs (Similar with that of inference process). The expectation of attribute vertices with attribute type $A_j$ is defined as

$$E_{A_j}=\frac{1}{\sum_i \deg(v_i)}\sum_i v_i \times \deg(v_i)$$

Then the utility is defined as

$$U_E = \delta_E(E_G,E_{G*})$$

### Edge weight

The edge weight can be determined by the similarity between two ends (social nodes). The higher weight indicates higher importance. We want to publish social relations with high weight without exposing the detailed relationship between two nodes (e.g. colleague: having the same employer). Therefore, there exists a trade-off between loss of important edges and privacy protection.




```python

```
