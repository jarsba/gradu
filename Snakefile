import sys

from src.utils.snakemake_utils import query_dataset_product, generate_products, generate_dicretization_product, generate_independence_pruning_products, query_independence_pruning_product

configfile: "config.yaml"
workdir: config['workdir']
sys.path.append(config['workdir'])

n_synt_datasets = config['n_synt_datasets']
epsilons = config['epsilons']
MCMC_algorithms = config['MCMC_algorithms']
queries = config['queries']
la_approx = ['LA', 'NoLA']
discretization_levels = ['low', 'high']

dataset = config['datasets']
dataset_names = [key for key in dataset.keys()]
dataset_files = [value for value in dataset.values()]

original_datasets = config['original_datasets']
original_dataset_names = [key for key in original_datasets.keys()]
original_dataset_files = [value for value in original_datasets.values()]

queries_dataset_product = query_dataset_product(original_datasets,queries)
experiment_products = generate_products(queries_dataset_product,epsilons,MCMC_algorithms)

discretization_datasets = config['discretization_datasets']
discretization_dataset_names = [key for key in discretization_datasets.keys()]
discretization_dataset_files = [value for value in discretization_datasets.values()]

discretization_products = generate_dicretization_product(epsilons, discretization_levels)

independence_pruning_datasets = config['independence_pruning_datasets']
independence_pruning_dataset_names = [key for key in independence_pruning_datasets.keys()]
independence_pruning_dataset_files = [value for value in independence_pruning_datasets.values()]

independence_pruning_queries = query_independence_pruning_product()
independence_pruning_products = generate_independence_pruning_products(epsilons, independence_pruning_queries)

singularity: "docker://continuumio/miniconda3:4.12.0"

wildcard_constraints:
    experiment_id="[a-zA-Z\d]{8}"

localrules: csv_results

rule csv_results:
    input:
        "results/original_logistic_regression_results.csv",
        "results/synthetic_logistic_regression_results.csv",
        "results/original_classification_results.csv",
        "results/synthetic_classification_results.csv",
        "results/discretization_logistic_regression_results.csv",
        "results/independence_pruning_logistic_regression_results.csv",
        "results/ci_coverage_original_data_results.csv",
        "results/ci_coverage_discretized_data_results.csv",
        "results/ci_coverage_independence_pruning_results.csv"


rule generate_original_datasets:
    output:
        expand("{dataset_file}",dataset_file=dataset_files)
    log:
        expand("logs/original_data_generation_dataset_{dataset_name}.log",dataset_name=dataset_names)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_original_datasets.py"


rule create_original_napsu_models:
    input:
        expand("{dataset_file}",dataset_file=original_dataset_files)
    output:
        expand("models/napsu_{experiment_product}.dill",experiment_product=experiment_products),
        "napsu_MCMC_time_vs_epsilon_comparison.csv",
        "napsu_experiment_storage_output.csv"
    log:
        expand("logs/napsu_{experiment_product}.log",experiment_product=experiment_products),
        expand("logs/inf_data_{experiment_product}.nc",experiment_product=experiment_products)
    threads: 8
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=48000,
        partition="medium"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/create_original_napsu_models.py"


rule generate_synthetic_datasets:
    input:
        expand("models/napsu_{experiment_product}.dill",experiment_product=experiment_products)
    output:
        expand("data/synt_datasets/synthetic_dataset_{experiment_product}.pickle",experiment_product=experiment_products)
    log:
        expand("logs/data_generation_synthetic_dataset_{experiment_product}.log",experiment_product=experiment_products)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_synthetic_datasets.py"


rule create_models_for_independence_pruning:
    input:
        expand("{dataset_file}",dataset_file=independence_pruning_dataset_files)
    output:
        expand("models/napsu_independence_pruning_{independence_pruning_product}.dill", independence_pruning_product=independence_pruning_products)
    log:
        expand("logs/napsu_independence_pruning_{independence_pruning_product}.log", independence_pruning_product=independence_pruning_products),
        expand("logs/inf_data_independence_pruning_{independence_pruning_product}.nc", independence_pruning_product=independence_pruning_products)
    threads: 8
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=48000,
        partition="medium"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/create_models_for_independence_pruning.py"


rule generate_datasets_for_independence_pruning:
    input:
        expand("models/napsu_independence_pruning_{independence_pruning_product}.dill", independence_pruning_product=independence_pruning_products)
    output:
        expand("data/synt_datasets/synthetic_dataset_independence_pruning_{independence_pruning_product}.pickle", independence_pruning_product=independence_pruning_products)
    log:
        expand("logs/data_generation_independence_pruning_{independence_pruning_product}.log", independence_pruning_product=independence_pruning_products)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_independence_pruning_datasets.py"

rule create_models_for_discretization:
    input:
        expand("{dataset_file}",dataset_file=discretization_dataset_files)
    output:
        expand("models/napsu_discretization_{discretization_product}.dill", discretization_product=discretization_products)
    log:
        expand("logs/inf_data_discretization_{discretization_product}.nc", discretization_product=discretization_products),
        expand("logs/napsu_discretization_{discretization_product}.log", discretization_product=discretization_products)
    threads: 8
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=48000,
        partition="medium"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/create_models_with_discretized_data.py"


rule generate_datasets_for_discretized_data:
    input:
        expand("models/napsu_discretization_{discretization_product}.dill", discretization_product=discretization_products)
    output:
        expand("data/synt_datasets/synthetic_dataset_discretization_{discretization_product}.pickle", discretization_product=discretization_products)
    log:
        expand("logs/data_generation_discretization_{discretization_product}.log", discretization_product=discretization_products)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_datasets_for_discretized_data.py"


rule run_logistic_regression_on_synt:
    input:
        expand("data/synt_datasets/synthetic_dataset_{experiment_product}.pickle",experiment_product=experiment_products)
    log:
        expand("logs/logistic_regression_synthetic_dataset_{experiment_product}.log",experiment_product=experiment_products)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=32000,
        disk_mb=50000,
        partition="short"
    output:
        "results/synthetic_logistic_regression_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr_on_synth.py"


rule run_logistic_regression_on_original:
    input:
        expand("{dataset_file}", dataset_file=original_dataset_files)
    output:
        "results/original_logistic_regression_results.csv"
    log:
        expand("logs/logistic_regression_original_dataset_{dataset}.log",dataset=original_datasets)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=32000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr_on_original.py"


rule run_classification_on_synt:
    input:
        expand("data/synt_datasets/synthetic_dataset_{experiment_product}.pickle",experiment_product=experiment_products)
    output:
        "results/synthetic_classification_results.csv"
    log:
        expand("logs/classification_synthetic_dataset_{experiment_product}.log",experiment_product=experiment_products)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=32000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_clf_on_synth.py"


rule run_classification_on_original:
    input:
        expand("{dataset_file}", dataset_file=original_dataset_files)
    output:
        "results/original_classification_results.csv"
    log:
        expand("logs/classification_original_dataset_{dataset}.log",dataset=original_datasets)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=32000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_clf_on_original.py"


rule run_logistic_regression_on_discretized_data:
    input:
        expand("data/synt_datasets/synthetic_dataset_discretization_{discretization_product}.pickle", discretization_product=discretization_products)
    output:
        "results/discretization_logistic_regression_results.csv"
    log:
        expand("logs/logistic_regression_discretized_dataset_{discretization_product}.log", discretization_product=discretization_products)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=32000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr_on_synth.py"

rule run_logistic_regression_on_independence_pruning_datasets:
    input:
        expand("data/synt_datasets/synthetic_dataset_independence_pruning_{independence_pruning_product}.pickle", independence_pruning_product=independence_pruning_products)
    output:
        "results/independence_pruning_logistic_regression_results.csv"
    log:
        expand("logs/logistic_regression_independence_pruning_dataset_{independence_pruning_product}.log",independence_pruning_product=independence_pruning_products)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=32000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr_on_synth.py"


rule run_ci_coverage_on_original_models:
    input:
        expand("models/napsu_{experiment_product}.dill",experiment_product=experiment_products)
    output:
        "results/ci_coverage_original_data_results.csv"
    log:
        expand("logs/ci_coverage_original_models_{experiment_product}.log",experiment_product=experiment_products)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_ci_coverage_on_original_models.py"


rule run_ci_coverage_on_discretized_models:
    input:
        expand("models/napsu_discretization_{discretization_product}.dill", discretization_product=discretization_products)
    output:
        "results/ci_coverage_discretized_data_results.csv"
    log:
        expand("logs/ci_coverage_discretization_{discretization_product}.log", discretization_product=discretization_products)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_ci_coverage_on_discretized_models.py"

rule run_ci_coverage_on_independence_pruning_models:
    input:
        expand("models/napsu_independence_pruning_{independence_pruning_product}.dill", independence_pruning_product=independence_pruning_products)
    output:
        "results/ci_coverage_independence_pruning_results.csv"
    log:
        expand("logs/ci_coverage_independence_pruning_{independence_pruning_product}.log", independence_pruning_product=independence_pruning_products)
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_ci_coverage_on_independence_pruning_models.py"


rule compare_original_lr_results:
    input:
        "results/synthetic_logistic_regression_results.csv",
        "results/original_logistic_regression_results.csv"
    output:
        report(expand("plots/lr_comparison_{dataset}.svg",dataset=original_datasets))
    log:
        "logs/comparison_logistic_regression.log"
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/plot_lr_results.py"


rule compare_original_clf_results:
    input:
        "results/synthetic_classification_results.csv",
        "results/original_classification_results.csv"
    output:
        report(expand("plots/clf_comparison_{dataset}.svg",dataset=original_datasets))
    log:
        "logs/comparison_classification.log"
    threads: 1
    resources:
        runtime="120",
        time="02:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/plot_clf_results.py"


rule clean_slurm_logs:
    shell:
        "rm slurm_logs/*"
