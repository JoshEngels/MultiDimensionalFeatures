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

Unfortunately, we did not set a seed when we ran spectral clustering in our original experiments, so the clusters you get from the above command may not be the same as the ones we used in the paper. In the `sae_multid_feature_discovery` directory, we provide the GPT-2 (`gpt-2_layer_7_clusters_spectral_n1000.pkl`) and Mistral-7B (`mistral_layer_8_clusters_cutoff_0.5.pkl`) clusters that were used in the paper. For easy reference, here are the GPT-2 SAE feature indices for the days, weeks, and years clusters we reported in the paper (Figure 1):

- Days of week: `[2592, 4445, 4663, 4733, 6531, 8179, 9566, 20927, 24185]`
- Months of year: `[3977, 4140, 5993, 7299, 9104, 9401, 10449, 11196, 12661, 14715, 17068, 17528, 19589, 21033, 22043, 23304]`
- Years of 20th century: `[1052, 2753, 4427, 6382, 8314, 9576, 9606, 13551, 19734, 20349]`

As a quick sanity check, the average pairwise cosine sim between decoder vectors for these clusters should be high (0.63, 0.53, and 0.55 respectively, ignoring the self-similarities).

To create an interactive cluster reconstruction plot for GPT-2, run
```
python3 gpt2_interactive_figure.py --cluster 138
```
This will produce a plotly html file and a corresponding png file.

To create an interactive cluster reconstruction plot for Mistral 7B, run
```
python3 mistral_interactive_figure.py --cluster 61
```
These will save an html file and a png file to directories within the `sae_multid_feature_discovery` folder. The html can be opened and interacted with in a browser. Mousing over a point will show the context and particular token (in bold) that the representation fired above.

To reproduce exactly Figure 1 and Figure 12, showing GPT-2 days, weeks, and years representations, you can run:
```
python3 gpt2_days_weeks_years.py
```

To produce Figure 13, showing Mistral 7b days and months representations, you can run:
```
python3 mistral_days_months.py
```


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
@article{engels2024not,
  title={Not All Language Model Features Are Linear},
  author={Engels, Joshua and Liao, Isaac and Michaud, Eric J and Gurnee, Wes and Tegmark, Max},
  journal={arXiv preprint arXiv:2405.14860},
  year={2024}
}
```


