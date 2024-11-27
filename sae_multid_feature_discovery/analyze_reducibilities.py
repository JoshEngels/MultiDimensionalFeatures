# %%
import json
import matplotlib.pyplot as plt
import numpy as np

all_reducibilities = []

# for threshold, radius in [(0, 0), (2, 2), (5, 0), (5, 1), (5, 3)]:
for threshold, radius in [(0, 0)]:

    base_dir = f"/media/jengels/sda/reducibility-metrics/threshold{threshold}radius{radius}"

    cluster_metas = []
    for cluster_id in range(1000):
        try:
            metrics_file = f"{base_dir}/{cluster_id}/gpt2-layer7-cluster{cluster_id}-threshold{float(threshold)}-radius{float(radius)}-metrics.json"
            data = json.load(open(metrics_file))
            cluster_metas.append(data)
        except:
            cluster_metas.append(None)

    mixure_bottom_floor = 0.01

    good_cluster_ids_ungrouped = []
    good_cluster_values_ungrouped = []
    for cluster_id, cluster_meta in enumerate(cluster_metas):
        if cluster_meta is None:
            continue
        for key, metrics in cluster_meta.items():
            if metrics["mixture_index"] < mixure_bottom_floor:
                continue
            good_cluster_ids_ungrouped.append((cluster_id, key))
            good_cluster_values_ungrouped.append((metrics["mixture_index"], metrics["separability_index"]))        

    good_cluster_ids_grouped = {}
    for i in range(len(good_cluster_ids_ungrouped)):
        cluster_id, key = good_cluster_ids_ungrouped[i]
        if cluster_id not in good_cluster_ids_grouped:
            good_cluster_ids_grouped[cluster_id] = ([], [])
        good_cluster_ids_grouped[cluster_id][0].append(good_cluster_values_ungrouped[i][0])
        good_cluster_ids_grouped[cluster_id][1].append(good_cluster_values_ungrouped[i][1])

    # Take mean of each cluster
    good_cluster_ids = []
    good_cluster_values = []
    good_cluster_id_to_value = {}
    for cluster_id, (mixture_indices, separability_indices) in good_cluster_ids_grouped.items():
        good_cluster_ids.append(cluster_id)
        good_cluster_values.append((np.mean(mixture_indices), np.mean(separability_indices)))
        good_cluster_id_to_value[cluster_id] = (np.mean(mixture_indices), np.mean(separability_indices))

    plt.scatter(*zip(*good_cluster_values), color="grey", alpha=0.3)
    plt.xlabel("Mixture Index")
    plt.ylabel("Separability Index")
    plt.gca().invert_xaxis()
    plt.title("GPT-2 Mixture Index vs Separability Index")

    special_to_plot = [138, 251, 212]
    names_to_plot = ["Weekdays cluster", "Months cluster", "Years cluster"]
    
    for cluster_id in good_cluster_ids_grouped:
        if cluster_id not in special_to_plot:
            continue
        cluster_value = good_cluster_id_to_value[cluster_id]
        name = names_to_plot[special_to_plot.index(cluster_id)]
        if cluster_id == 251:
            plt.text(cluster_value[0] + 0.13, cluster_value[1] + 0.02, name, fontsize=14)
        elif cluster_id == 212:
            plt.text(cluster_value[0] + 0.01, cluster_value[1] - 0.05, name, fontsize=14)
        else:
            plt.text(cluster_value[0] + 0.02, cluster_value[1] + 0.01, name, fontsize=14)
        plt.scatter(*cluster_value, color="orange")

    # Save fig
    plt.savefig(f"reducibility_metrics_threshold{threshold}_radius{radius}.pdf", bbox_inches="tight")

    # Rank clusters by max (1 - mixture_index) * separability_index)
    cluster_values = {i: [0] for i in range(1000)}
    for cluster_id, key in good_cluster_ids_ungrouped:
        cluster_values[cluster_id].append(
            (1 - cluster_metas[cluster_id][key]["mixture_index"])
            * cluster_metas[cluster_id][key]["separability_index"],
        )
    for key in cluster_values:
        cluster_values[key] = np.mean(cluster_values[key])

    # Sort cluster ids by cluster values
    cluster_ids = range(1000)
    cluster_ids = sorted(cluster_ids, key=lambda x: cluster_values[x], reverse=True)
    print([cluster_ids.index(i) for i in special_to_plot])

    print(cluster_ids[:20])

# %%
