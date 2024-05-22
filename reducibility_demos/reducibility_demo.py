import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

plt.rcParams["text.usetex"] = True

for dataset_num in range(1, 5):
    np.random.seed(2)
    torch.manual_seed(2)

    if dataset_num == 3:
        plt.rcParams.update({"font.size": 30})
    else:
        plt.rcParams.update({"font.size": 30})

    ### DATASET CONSTRUCTION
    batch_size = 4096

    def pick_fn(dataset_num):
        n = 2
        randmat = torch.randn([n, n])
        randvect1 = torch.randn([n])
        randoffset1 = torch.randn([n])
        randvect2 = torch.randn([n])
        randoffset2 = torch.randn([n])

        if dataset_num == 1:
            x = torch.tensordot(torch.randn([batch_size, n]), randmat, dims=1)
        if dataset_num == 2:
            a = torch.randn([batch_size, n])
            b = torch.randn([batch_size, 1]) * randvect1 + randoffset1
            c = torch.randn([batch_size, 1]) * randvect2 + randoffset2
            picker = torch.rand([batch_size, 1])
            option_1 = (picker < 0.4).type(a.dtype)
            option_2 = (picker < 0.7).type(a.dtype) - option_1
            option_3 = (picker < 1).type(a.dtype) - option_2 - option_1
            x = a * option_1 + b * option_2 + c * option_3
        if dataset_num == 3:
            x = torch.randn([batch_size, n])
            norms = torch.sqrt(torch.sum(x**2, axis=1))
            x = x / norms[:, None]
        if dataset_num == 4:
            x = torch.randn([batch_size, n])
            x = torch.tensordot(
                (3 * x).type(torch.int32).type(x.dtype), randmat, dims=1
            )
        return n, x

    n, x = pick_fn(dataset_num)

    ### MIXTURE TESTING
    def get_concentration_probability(x, epsilon, temperature, a, b):
        x = torch.tensordot(x, a, dims=1) + b
        z = x / torch.sqrt(torch.mean(x**2))
        P = torch.mean(torch.sigmoid((epsilon - torch.abs(z)) / temperature))
        return P

    def get_parameters(x, epsilon=0.1):
        # Initialize the parameter x
        a = torch.randn(
            [n], requires_grad=True
        )  # Random initialization, requires_grad=True to track gradients
        b = torch.zeros(
            [], requires_grad=True
        )  # Random initialization, requires_grad=True to track gradients

        # Define hyperparameters
        learning_rate = 0.1
        num_iterations = 10000
        #    num_iterations = 100

        # Gradient Descent loop
        for i in range(num_iterations):
            temperature = 1 - i / num_iterations
            # Compute the function value and its gradient
            P = get_concentration_probability(x, epsilon, temperature, a, b)
            P.backward()  # Compute gradients
            with torch.no_grad():
                # Update x using gradient descent
                a += learning_rate * a.grad
                b += learning_rate * b.grad

            # Manually zero the gradients after updating weights
            a.grad.zero_()
            b.grad.zero_()
        return a.detach().numpy(), b.detach().numpy(), P.item()

    ### SEPARABILITY TESTING
    def bin_(xy, n, bound):
        xy = torch.clip((xy + bound) / bound * n, 0, 2 * n - 0.1)
        ints = torch.floor(xy).type(torch.int64)
        dist = torch.zeros([(2 * n) ** 2])
        indices = ints[:, 0] * (2 * n) + ints[:, 1]
        dist.scatter_add_(0, indices, torch.ones(indices.shape, dtype=xy.dtype))
        dist = torch.reshape(dist, [2 * n, 2 * n])

        #    mods = xy % 1
        #    weightings = 1 - mods
        #    dist = torch.zeros([(2*n+2)**2])
        #    indices = ints[:,0]*(2*n+2) + ints[:,1]
        #    dist.scatter_add_(0, indices, weightings[:,0]*weightings[:,1])
        #    indices = (ints[:,0]+1)*(2*n+2) + ints[:,1]
        #    dist.scatter_add_(0, indices, (1-weightings[:,0])*weightings[:,1])
        #    indices = ints[:,0]*(2*n+2) + (ints[:,1]+1)
        #    dist.scatter_add_(0, indices, weightings[:,0]*(1-weightings[:,1]))
        #    indices = (ints[:,0]+1)*(2*n+2) + (ints[:,1]+1)
        #    dist.scatter_add_(0, indices, (1-weightings[:,0])*(1-weightings[:,1]))
        #    dist = torch.reshape(dist, [2*n+2, 2*n+2])
        return dist

    def mutual_info(xy, n, bound):
        dist = bin_(xy, n, bound)
        joint = dist / torch.sum(dist)
        marginal_x = torch.sum(joint, dim=1)
        marginal_y = torch.sum(joint, dim=0)
        product = marginal_x[:, None] * marginal_y[None, :]
        mutual_info = torch.sum(joint * torch.log((joint + 1e-4) / (product + 1e-4)))
        return mutual_info

    def optimize_mutual_info(xy, split, n, bound, angular_res):  # xy: n, d
        d_x = split
        d_y = xy.shape[1] - split
        assert d_x == 1 and d_y == 1

        mutual_infos = []
        for i in range(angular_res):
            angle = torch.tensor(2 * np.pi * i / angular_res)
            rotation = torch.cos(angle).type(xy.dtype) * torch.eye(
                2, dtype=xy.dtype
            ) + torch.sin(angle).type(xy.dtype) * torch.tensor(
                [[0, -1], [1, 0]], dtype=xy.dtype
            )
            minfo = mutual_info(torch.tensordot(xy, rotation, dims=1), n, bound)
            mutual_infos.append(minfo)

        max_ind = torch.argmin(torch.tensor(mutual_infos), dim=0)
        angle = 2 * np.pi * max_ind / angular_res
        rotation = torch.cos(angle).type(xy.dtype) * torch.eye(
            2, dtype=xy.dtype
        ) + torch.sin(angle).type(xy.dtype) * torch.tensor(
            [[0, -1], [1, 0]], dtype=xy.dtype
        )
        rot_xy = torch.tensordot(xy, rotation, dims=1)
        return mutual_infos, max_ind, rotation

    def normalize(x):
        x = x - torch.mean(x, dim=0)
        x = x / torch.mean(x**2)
        return x

    fig = plt.figure(figsize=(24, 5))
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.03, 0.92, 0.05])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])
    axs = [ax1, ax2, ax3]

    # Mixture testing
    epsilon = 0.1
    a, b, P = get_parameters(x, epsilon)

    x_numpy = x.numpy()

    a_norm = np.sqrt(np.sum(a**2))
    normalized_a = a / a_norm

    proj_x = (np.tensordot(x_numpy, a, axes=1) + b) / a_norm
    eps = epsilon * np.sqrt(np.mean(proj_x**2))

    z = proj_x / np.sqrt(np.mean(proj_x**2))
    axs[1].hist(z, bins=100, color="k")
    axs[1].axvline(x=-epsilon, color="red", linestyle=(0, (5, 5)))
    axs[1].axvline(x=epsilon, color="red", linestyle=(0, (5, 5)))
    axs[1].set_xlabel("normalized $\\mathbf{v} \\cdot \\mathbf{f} + c$")
    axs[1].set_ylabel("count")
    if dataset_num == 1:
        height = 65
        xpos = -3.8
    if dataset_num == 2:
        height = 500
        xpos = 0.4
    if dataset_num == 3:
        height = 150
        xpos = -1.1
    if dataset_num == 4:
        height = 700
        xpos = 0.4
    axs[1].text(
        xpos,
        height,
        "$M_\\epsilon(\\mathbf{f})=" + str(round(float(P), 4)) + "$",
        color="red",
    )
    axs[1].spines[["top", "left", "right"]].set_visible(False)
    axs[1].grid(axis="y")

    b_norm = b / a_norm
    b = normalized_a * b_norm

    axs[0].scatter(x_numpy[:, 0], x_numpy[:, 1], color="k", s=2)
    axs[0].axline(
        -b + eps * normalized_a[:2],
        slope=-a[0] / a[1],
        color="red",
        linestyle=(0, (5, 5)),
    )
    axs[0].axline(
        -b - eps * normalized_a[:2],
        slope=-a[0] / a[1],
        color="red",
        linestyle=(0, (5, 5)),
    )

    axs[0].axis("equal")
    axs[0].set_xlabel("representation dim 1")
    axs[0].set_ylabel("representation dim 2")

    print("Mixture index: ", P)

    # Separability testing
    x = normalize(x)
    xy = x
    angular_res = 1000
    n = 20
    bound = 3
    mutual_infos, max_ind, net_transform = optimize_mutual_info(
        xy, 1, n, bound, angular_res
    )
    mutual_info = mutual_infos[max_ind]

    dist = bin_(xy, n, bound)

    # axs[0].scatter(xy[:,0], xy[:,1], color='k', s=2)
    inv_transform = np.linalg.inv(net_transform.numpy())
    cross_size = [3, 4, 1.2, 5][dataset_num - 1]
    dir_x = cross_size * inv_transform[0, :]
    dir_y = cross_size * inv_transform[1, :]
    axs[0].plot([-dir_x[0], dir_x[0]], [-dir_x[1], dir_x[1]], "g")
    axs[0].plot([-dir_y[0], dir_y[0]], [-dir_y[1], dir_y[1]], "g")

    print("Separability index: ", mutual_info)

    axs[0].axis("equal")
    axs[0].set_xlabel("representation dim 1")
    axs[0].set_ylabel("representation dim 2")

    axs[0].spines[["left", "bottom", "right", "top"]].set_visible(False)
    axs[0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[0].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    axs[2].plot(
        2 * np.pi * np.arange(angular_res) / angular_res,
        [minfo.numpy() / np.log(2) for minfo in mutual_infos],
        "g",
    )
    offset = 0.0 if max_ind / angular_res % 0.25 > 0.1 else 0.25
    axs[2].plot(
        [
            2 * np.pi * (max_ind / angular_res % 0.25 + offset),
            2 * np.pi * (max_ind / angular_res % 0.25 + offset),
        ],
        [0, mutual_info / np.log(2)],
        "g--",
    )

    axs[2].text(
        2 * np.pi * (max_ind / angular_res % 0.25 + offset) + 0.1,
        max(0.05, mutual_info / np.log(2) - 0.3),
        "$S(\\mathbf{f})=" + str(round(float(mutual_info / np.log(2)), 4)) + "$",
        color="green",
    )
    axs[2].set_xlabel("angle $\\theta$")
    axs[2].set_ylabel("Mutual info (bits)")

    axs[2].spines[["top"]].set_visible(False)
    axs[2].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    axs[2].tick_params(axis="y", which="both", left=True, right=False, labelleft=True)
    axs[2].grid(axis="y")

    # set the x-spine (see below for more info on `set_position`)
    axs[2].spines["left"].set_position(("data", 0))
    axs[2].spines["right"].set_position(("data", 2 * np.pi))
    axs[2].spines["bottom"].set_position(("data", 0))

    axs[2].set_xlim((0, 2 * np.pi))
    axs[2].set_ylim((0, np.max([minfo.numpy() / np.log(2) for minfo in mutual_infos])))

    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    axs[2].set_xticks(ticks)
    labels = ["0", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$", "$2\\pi$"]
    axs[2].set_xticklabels(labels)

    dataset_name = ["gaussian", "gaussians", "circle", "lattice"][dataset_num - 1]
    plt.savefig(
        "reducibility_"
        + dataset_name
        + "_"
        + str(round(float(P), 4))
        + "_"
        + str(round(float(mutual_info.numpy()), 4))
        + ".pdf",
        bbox_inches="tight",
    )
    plt.close()
