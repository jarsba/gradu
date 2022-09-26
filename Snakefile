configfile: "config.yaml"

datasets = config['datasets']
epsilons = config['epsilons']
MCMC_algorithms = config['MCMC_algorithms']

napsu_results = expand("_{epsilon}e_{MCMC_algorithm}", epsilon=epsilons, MCMC_algorithm=MCMC_algorithms)
synt_data_results = expand("synt_data_{dataset}_{epsilon}e_{MCMC_algorithm}", dataset=datasets, epsilon=epsilons, MCMC_algorithm=MCMC_algorithms)


rule all:
    input:
        "plots/{lr_comparison}.svg",
        "plots/{clf_comparison}.svg"


rule run_napsu:
    input:
        "data/datasets/{dataset}.csv"
    output:
        "models/napsu_{dataset}_{napsu_results}.npy"
    log:
        "logs/napsu/napsu_{dataset}_{napsu_results}.log"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/run_napsu.py"


rule generate_synt_datasets:
    input:
        "models/{napsu_results}.npy"
    output:
        "data/synt_datasets/synthetic_dataset_{name}_{index}_{parameters}_{id}.csv",
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/run_napsu.py"


rule run_logistic_regression_on_synt:
    input:
        "data/synt_datasets/{synt_dataset}.csv"
    output:
        "results/synthetic_logistic_regression_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr.py"


rule run_logistic_regression_on_original:
    input:
        "data/datasets/{dataset}.csv"
    output:
        "results/original_logistic_regression_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr.py"


rule run_classification_on_synt:
    input:
        "data/synt_datasets/{synt_dataset}.csv"
    output:
        "results/synthetic_classification_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_clf.py"


rule run_classification_on_original:
    input:
        "data/datasets/{dataset}.csv"
    output:
        "results/original_classification_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_clf.py"


rule compare_datasets:
    input:
        "data/datasets/{dataset}.csv",
        "data/synt_datasets/synthetic_dataset_{name}_{index}_{parameters}_{id}.csv"
    output:
        "results/comparison_{dataset}_{}.txt"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/compare_datasets.py"


rule compare_lr_results:
    input:
        "results/{synt_lr_result}.txt",
        "results/{orig_lr_result}.txt",
    output:
        report("plots/{lr_comparison}.svg"),
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/plot_results.py"


rule compare_clf_results:
    input:
        "results/{synt_clf_result}.txt",
        "results/{orig_clf_result}.txt"
    output:
        report("plots/{clf_comparison}.svg")
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/plot_results.py"
