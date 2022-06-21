# Latent-Space-Hawkes
Python implementation of the Latent Space Hawkes (LSH) model for continuous-time dynamic networks

This repo contains the Python implementation and code to replicate experiments for LSH model in the paper [A Mutually Exciting Latent Space Hawkes Process Model for Continuous-time Networks](https://arxiv.org/abs/2205.09263), accpeted by UAI 2022.

---

## Abstract

Networks and temporal point processes serve as fundamental building blocks for modeling complex dynamic relational data in various domains. We propose the latent space Hawkes (LSH) model, a novel generative model for continuous-time networks of relational events, using a latent space representation for nodes. We model relational events
between nodes using mutually exciting Hawkes processes with baseline intensities dependent upon the distances between the nodes in the latent space and sender and receiver specific effects. We propose an alternating minimization algorithm to jointly estimate the latent positions of the nodes
and other model parameters. We demonstrate that our proposed LSH model can replicate many features observed in real temporal networks including reciprocity and transitivity, while also achieves superior prediction accuracy and provides more interpretability compared to existing models.

---

## Running the experiments

### Requirements

Dependencies (with python >= 3.7)

* numpy==1.18.5
* scikit-learn==0.23.1
* matplotlib==3.2.2
* networkx==2.4
* dynetworkx==0.3.2
* seaborn==0.10.1
* scipy==1.5.0
* autograd=1.3
* pickle=4.0


### Datasets and results
enron-events.pckl: stores the Enron data provided in [Yang's github-repo](https://github.com/jiaseny/lspp). All other datasets are available in the storage/datasets; All results will be stored in the storage/results

### Model running

#### Future events prediction tasks:

```python
# run the model on reality mining dataset with a 2 dimendion LSH model
python LSH_model_fit.py --dataset 'reailty' --dim 2
```

#### Dynamic link prediction tasks:

```python
# run dynamic link prediction on reality mining dataset with a 2 dimendion model
python dynamic_link_pred.py --dataset 'reailty' --dim 2
```

#### Genetative tasks (Posterior predictive check):

```python
# run the generative test on reality mining dataset with a 2 dimendion model
python LSH_generative.py --dataset 'reailty' --dim 2
```

#### Simulation tasks

```python
# run the simulation experiments
python LSH_model_sim.py
```

#### General flags
```
optional arguments:
--data   Dataset to use (eg. reality, Enron-Yang, MID, fb-forum)
-d --dim  Latent dimensions for LSH model
-c -- continent  Whether to plot 2D MID latent space with continent colored
```
