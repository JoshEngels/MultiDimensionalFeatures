# Multi-Dimensional Features
Code for reproducing our paper "Not All Language Model Features Are Linear"


## Reproducing each figure

Below are instructions to reproduce each figure. 

All needed dependencies are in requirements.txt; we recommend you create a new conda environment with these libraries, e.g.:
```
conda create --name multid_features --file requirements.txt
```
This may be a superset of needed dependencies, so feel free to create a fresh environment and only install libraries as needed.

### Intervention Experiments

Before running experiments, you should change BASE_DIR in intervention/utils.py to point to a location on your machine where large artifacts can be downloaded and saved (Mistral and Llama 3 take ~60GB and experiment artifacts are ~100GB).

To reproduce the intervention results, you will first need to run intervention experiments with the following commands:

```
cd intervention
python3 circle_probe_interventions.py day a mistral --device 0 --intervention_pca_k 5 --probe_on_cos --probe_on_sin
python3 circle_probe_interventions.py month a mistral --device 0 --intervention_pca_k 5 --probe_on_cos --probe_on_sin
python3 circle_probe_interventions.py day a llama --device 0 --intervention_pca_k 5 --probe_on_cos --probe_on_sin
python3 circle_probe_interventions.py month a llama --device 0 --intervention_pca_k 5 --probe_on_cos --probe_on_sin
```

You can then reproduce *Figure 3*, *Figure 5*, *Figure 6*, and *Table 1* by running the corresponding cells in intervention/main_text_plots.ipynb.


After running these intervention experiments, you can reproduce *Figure 6* by running 
```
cd intervention
python3 intervene_in_middle_of_circle.py --only_paper_plots
```
and then running the corresponding cell in intervention/main_text_plots.ipynb.

You can reproduce *Figure 14*, *Figure 15*, *Figure 16*, *Table 2*, *Table 3*, and *Table 4* (all from the appendix) by running cells in intervention/appendix_plots.ipynb.


### SAE feature search experiments

Before running experiments, you should again change BASE_DIR in intervention/utils.py to point to a location on your machine where large artifacts can be downloaded and saved.

You will first need to generate SAE feature activations for clustering (the current hyperparameters in the file work for GPT-2 and automatically download pretrained SAEs for GPT-2):

```
cd sae_multi_feature_discovery
python3 generate_feature_occurence_data.py
```

