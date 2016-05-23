# Learning to Discover Social Circles in Ego Networks

## Introduction

Goal: automactically discovering users' social circles

Problem Formulation: Circle detection as clustering problem given edge set and common properties

Method: unsupervised method to learn which dimension of profile similarity lead to densely linked circles

## Generative Model For Friendships in Ego Networks 

Circule Formation:

1. Nodes within circles should have common properties

2. Different circles should be formed by different aspects

3. Circles are allowed to overlap

4. It is necessary to leverage both profile information and network structure

Model Description

ego-network G=(V,E), and center node is not included in G. The goal is to predict a set of circles C, and associated parameter vectors 


