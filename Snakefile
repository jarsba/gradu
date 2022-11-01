import sys

from src.utils.snakemake_utils import query_dataset_product, generate_products

configfile: "config.yaml"
workdir: config['workdir']
sys.path.append(config['workdir'])

n_synt_datasets = config['n_synt_datasets']
dataset = config['datasets']
dataset_names = [key for key in dataset.keys()]
dataset_files = [value for value in dataset.values()]
epsilons = config['epsilons']
MCMC_algorithms = config['MCMC_algorithms']
queries = config['queries']
la_approx = ['LA', 'NoLA']

queries_dataset_product = query_dataset_product(dataset_names, queries)
experiment_products = generate_products(queries_dataset_product, epsilons, MCMC_algorithms)


wildcard_constraints:
    experiment_id="[a-zA-Z\d]{8}"

localrules: all, csv_results

rule all:
    input:
        expand("plots/lr_comparison_{dataset}.svg",dataset=dataset_names),
        expand("plots/clf_comparison_{dataset}.svg",dataset=dataset_names)

rule csv_results:
    input:
        "results/original_logistic_regression_results.csv",
        "results/synthetic_logistic_regression_results.csv",
        "results/original_classification_results.csv",
        "results/synthetic_classification_results.csv"



rule run_napsu:
    input:
        expand("{dataset_file}", dataset_file=dataset_files)
    output:
        expand("models/napsu_{experiment_product}.dill", experiment_product=experiment_products),
    #"napsu_MCMC_time_vs_epsilon_comparison.csv",
    #"napsu_experiment_storage_output.csv"
    log:
        expand("logs/napsu_{experiment_product}.log", experiment_product=experiment_products)
    threads: 4
    resources:
        runtime = "2160",
        time = "36:00:00",
        mem_mb = 16000,
        partition = "medium"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/run_napsu.py"


rule generate_synt_datasets:
    input:
        expand("models/napsu_{experiment_product}.dill", experiment_product=experiment_products)
    output:
        expand("data/synt_datasets/synthetic_dataset_{experiment_product}.pickle", experiment_product=experiment_products)
    log:
        expand("logs/data_generation_synthetic_dataset_{experiment_product}.log", experiment_product=experiment_products)
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_datasets.py"


rule run_logistic_regression_on_synt:
    input:
        expand("data/synt_datasets/synthetic_dataset_{experiment_product}.pickle", experiment_product=experiment_products)
    log:
        expand("logs/logistic_regression_synthetic_dataset_{experiment_product}.log", experiment_product=experiment_products)
    output:
        "results/synthetic_logistic_regression_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr_on_synth.py"


rule run_logistic_regression_on_original:
    input:
        expand("{dataset_file}",dataset_file=dataset_files)
    output:
        "results/original_logistic_regression_results.csv"
    log:
        expand("logs/logistic_regression_original_dataset_{dataset}.log", dataset=dataset_names)
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr_on_synth.py"


rule run_classification_on_synt:
    input:
        expand("data/synt_datasets/synthetic_dataset_{experiment_product}.pickle", experiment_product=experiment_products)
    output:
        "results/synthetic_classification_results.csv"
    log:
        expand("logs/classification_synthetic_dataset_{experiment_product}.log", experiment_product=experiment_products)
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_clf_on_synth.py"


rule run_classification_on_original:
    input:
        expand("{dataset_file}",dataset_file=dataset_files)
    output:
        "results/original_classification_results.csv"
    log:
        expand("logs/classification_original_dataset_{dataset}.log", dataset=dataset_names)
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_clf_on_synth.py"


#rule compare_datasets:
#    input:
#        expand("data/datasets/{dataset}", dataset=dataset_files),
#        expand("data/synt_datasets/synthetic_dataset_{experiment_product}.csv", experiment_product=experiment_products)
#    output:
#        "results/comparison_{dataset}_vs_{parameters}.csv"
#    conda:
#        "envs/analysis.yaml"
#    script:
#        "scripts/compare_datasets.py"


rule compare_lr_results:
    input:
        "results/synthetic_logistic_regression_results.csv",
        "results/original_logistic_regression_results.csv",
    output:
        report(expand("plots/lr_comparison_{dataset}.svg",dataset=dataset_names))
    log:
        "logs/comparison_logistic_regression.log"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/plot_lr_results.py"


rule compare_clf_results:
    input:
        "results/synthetic_classification_results.csv",
        "results/original_classification_results.csv"
    output:
        report(expand("plots/clf_comparison_{dataset}.svg",dataset=dataset_names))
    log:
        "logs/comparison_classification.log"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/plot_clf_results.py"


rule clean_slurm_logs:
    shell:
        "rm slurm_logs/*"