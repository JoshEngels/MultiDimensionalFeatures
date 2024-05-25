<img width="1853" alt="GPT-2 Auto-Discovered Multi-Dimensional Features" src="https://github.com/JoshEngels/MultiDimensionalFeatures/assets/15754392/cbe67ac3-feed-41a2-b31f-2a75406030da">

# Multi-Dimensional Features
This is the github repo for our paper ["Not All Language Model Features Are Linear"](https://arxiv.org/abs/2405.14860).


## Reproducing each figure

Below are instructions to reproduce each figure (aspirationally). 

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

Before running experiments, you should again change BASE_DIR in sae_multid_feature_discovery/utils.py to point to a location on your machine where large artifacts can be downloaded and saved.

You will need to generate SAE feature activations to generate the cluster reconstructions. The GPT-2 SAEs will be automatically downloaded when you run the below scripts, while for Mistral you will need to download our pretrained Mistral SAEs from https://www.dropbox.com/scl/fo/hznwqj4fkqvpr7jtx9uxz/AJUe0wKmJS1-fD982PuHb5A?rlkey=ffnq6pm6syssf2p7t98q9kuh1&dl=0 to sae_multid_feature_discovery/saes/mistral_saes. You can generate SAE feature activations with one of the following two commands:

```
cd sae_multid_feature_discovery
python3 generate_feature_occurence_data.py --model_name gpt-2
python3 generate_feature_occurence_data.py --model_name mistral
```

You will also need to generate the actual clusters by running clustering.py, e.g.
```
python3 clustering.py --model_name gpt-2 --clustering_type spectral --layer 7
python3 clustering.py --model_name mistral --clustering_type graph --layer 8
```

To create an interactive cluster reconstruction plot for GPT-2 (TODO), run
```
python3 gpt2_figure.py --layer 7 --cluster 0
```
To create an interactive cluster reconstruction plot for Mistral 7B, run
```
python3 mistral_figure.py --layer 8 --cluster 0
```
These will save an html file and a png file to directories within the `sae_multid_feature_discovery` folder. The html can be opened and interacted with in a browser. Mousing over a point will show the context and particular token (in bold) that the representation fired above.

To make the final polished figures in the paper, we provide different scripts. To produce Figure 13, showing Mistral 7b days and months representations, you can run:
```
python3 mistral_days_months.py
```
We will add similar scripts for the GPT-2 figures soon.


### Reducibility Experiments

You can reproduce *Figure 2*, and *Figure 9* by running

```
cd reducibility_demos
python3 reducibility_demo.py
```

`reducibility_demo.py` is a self-contained code file which generates the synthetic datasets, computes the epsilon-mixture index, computes the separability index, and plots it all.


### Explanation via Regression with Residual RGB Plots

To reproduce the residual RGB plots in the paper (*Figure 8*, and *Figure 16*), you must first generate `results.csv` and a folder called `pca_components/` full of files named `layerX_tokenY_pca20.pt`. These files should all be in BASE_DIR after running the intervention_experiments above. `results.csv` lists out the addition problems given, and each `layerX_tokenY_pca20.pt` has an array that contains the top 20 PCA components of the hidden states outputted by layer X on token Y for each addition problem. To produce residual RGB plots for LLAMA 3 8B on the months of the year task, generate `results.csv` and `pca_components/` for LLAMA 3 8B, and copy them into `feature_deconstruction/months_of_the_year/`, `cd` to the directory `feature_deconstruction/months_of_the_year/`, and run `python3 months_of_the_year_deconstruction.py`. The same goes for Mistral 7B on the days of the week task, using `feature_deconstruction/days_of_the_week/`.


## Contact

If you have any questions about the paper or reproducing results, feel free to email jengels@mit.edu.

## Citation

```
@article{engels2024language,
      title={Not All Language Model Features Are Linear}, 
      author={Joshua Engels and Isaac Liao and Eric J. Michaud and Wes Gurnee and Max Tegmark},
      year={2024},
      journal={arXiv preprint arXiv:2405.14860}
}
```


