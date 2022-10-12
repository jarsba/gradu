from src.utils.preprocess_dataset import get_binary3d_train, get_adult_train, get_binary4d_train
from src.napsu_mq.main import create_model
from src.utils.timer import Timer

timer = Timer()

binary3d_data = get_binary3d_train()
binary4d_data = get_binary4d_train()
adult_data = get_adult_train()

datasets = {
    'binary_3d': binary3d_data,
    'binary_4d': binary4d_data,
    # 'adult': adult_data
}

epsilons = [0.01, 0.1, 1.0, 8.0]

dataset_to_cliques_map = {
    "binary_3d": [
        ('A', 'B'), ('B', 'C'), ('A', 'C')
    ],
    "binary_4d": [
        ('A', 'B'), ('B', 'C'), ('A', 'C'), ('A', 'D'), ('B', 'D')
    ],
    # "adult": [
    #    ("age", "compensation"), ("race", "compensation"), ("race", "sex"),
    #    ("age", "hours-per-week")
    # ]
}

# With Laplace approximation

for dataset_name, dataset in datasets.items():
    print(f"Testing {dataset_name} dataset with Laplace approximation")

    for epsilon in epsilons:
        print(f"Testing {dataset_name} dataset with epsilon {epsilon}")

        n, d = dataset.shape

        pid = timer.start(f"MCMC (LA) main run", dataset_name=dataset_name, epsilon=epsilon)

        model = create_model(
            input=dataset,
            dataset_name=dataset_name,
            epsilon=epsilon,
            delta=(n ** (-2)),
            cliques=dataset_to_cliques_map[dataset_name],
            MCMC_algo='NUTS'
        )

        timer.stop(pid)

# Without Laplace approximation

for dataset_name, dataset in datasets.items():
    print(f"Testing {dataset_name} dataset without Laplace approximation")

    for epsilon in epsilons:
        print(f"Testing {dataset_name} dataset with epsilon {epsilon}")

        n, d = dataset.shape

        pid = timer.start(f"MCMC (No LA) main run", dataset_name=dataset_name, epsilon=epsilon)

        model = create_model(
            input=dataset,
            dataset_name=dataset_name,
            epsilon=epsilon,
            delta=(n ** (-2)),
            cliques=dataset_to_cliques_map[dataset_name],
            MCMC_algo='NUTS',
            use_laplace_approximation=False
        )

        timer.stop(pid)

timer.to_csv("napsu_MCMC_time_vs_epsilon_comparison.csv")
