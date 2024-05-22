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

You can reproduce *Figure 13*, *Figure 14*, *Figure 15*, *Table 2*, *Table 3*, and *Table 4* (all from the appendix) by running cells in intervention/appendix_plots.ipynb.


### SAE feature search experiments

Before running experiments, you should again change BASE_DIR in intervention/utils.py to point to a location on your machine where large artifacts can be downloaded and saved.

You will need to generate SAE feature activations to generate the cluster reconstructions (the current hyperparameters in the file work for GPT-2 and automatically download pretrained SAEs for GPT-2; you can comment these out and uncomment out the ones for Mistral, in which case you will need to download our pretrained Mistral SAEs to sae_multid_feature_discovery/saes/mistral_saes):

```
cd sae_multid_feature_discovery
python3 generate_feature_occurence_data.py
```

You will also need to generate the actual clusters by running clustering.py:
```
python3 clustering.py --model_name [mistral, gpt_2] --clustering_type [spectral, graph]
```

TODO: Eric fill in how to generate cluster reconstructions and figures

### Reducibility Experiments

You can reproduce *Figure 2*, and *Figure 9* by running

```
cd reducibility_demos
python3 reducibility_demo.py
```

`reducibility_demo.py` is a self-contained code file which generates the synthetic datasets, computes the epsilon-mixture index, computes the separability index, and plots it all.


### Explanation via Regression with Residual RGB Plots

To reproduce the residual RGB plots in the paper (*Figure 8*, and *Figure 16*), you must first generate `results.csv` and a folder called `pca_components/` full of files named `layerX_tokenY_pca20.pt`. These files should all be in BASE_DIR after running the intervention_experiments above. `results.csv` lists out the addition problems given, and each `layerX_tokenY_pca20.pt` has an array that contains the top 20 PCA components of the hidden states outputted by layer X on token Y for each addition problem. To produce residual RGB plots for LLAMA 3 8B on the months of the year task, generate `results.csv` and `pca_components/` for LLAMA 3 8B, and copy them into `feature_deconstruction/months_of_the_year/`, `cd` to the directory `feature_deconstruction/months_of_the_year/`, and run `python3 months_of_the_year_deconstruction.py`. The same goes for Mistral 7B on the days of the week task, using `feature_deconstruction/days_of_the_week/`.
