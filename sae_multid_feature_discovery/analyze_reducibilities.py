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

    good_cluster_ids = []
    good_cluster_values = []
    for cluster_id, cluster_meta in enumerate(cluster_metas):
        if cluster_meta is None:
            continue
        for key, metrics in cluster_meta.items():
            if metrics["mixture_index"] < mixure_bottom_floor:
                continue
            good_cluster_ids.append((cluster_id, key))
            good_cluster_values.append((metrics["mixture_index"], metrics["separability_index"]))        


    plt.scatter(*zip(*good_cluster_values))
    plt.xlabel("Mixture Index")
    plt.ylabel("Separability Index")
    plt.gca().invert_xaxis()
    plt.title(f"Threshold {threshold}, Radius {radius}")

    # plot_in_orange = [67, 109, 134, 138, 157, 180, 212, 213, 223, 251, 285]
    plot_in_orange = [138, 157, 251]
    for cluster_id, key in good_cluster_ids:
        if cluster_id in plot_in_orange:
            plt.scatter(
                cluster_metas[cluster_id][key]["mixture_index"],
                cluster_metas[cluster_id][key]["separability_index"],
                color="orange",
            )
            plt.text(
                cluster_metas[cluster_id][key]["mixture_index"],
                cluster_metas[cluster_id][key]["separability_index"],
                f"{cluster_id}: {key}",
                fontsize=9,
            )

    plt.show()


    # Rank clusters by max (1 - mixture_index) * separability_index)
    cluster_values = {i: [0] for i in range(1000)}
    for cluster_id, key in good_cluster_ids:
        cluster_values[cluster_id].append(
            (1 - cluster_metas[cluster_id][key]["mixture_index"])
            * cluster_metas[cluster_id][key]["separability_index"],
        )
    for key in cluster_values:
        cluster_values[key] = np.mean(cluster_values[key])

    # Sort cluster ids by cluster values
    cluster_ids = range(1000)
    cluster_ids = sorted(cluster_ids, key=lambda x: cluster_values[x], reverse=True)
    print([cluster_ids.index(i) for i in plot_in_orange])

    print(cluster_ids[:20])
# %%
