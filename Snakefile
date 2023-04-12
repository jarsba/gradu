import sys
from src.utils.snakemake_utils import generate_dataset_query_epsilon_products, generate_independence_pruning_missing_queries, generate_dataset_query_epsilon_products_independence_queries, generate_linear_regression_products

configfile: "config.yaml"
workdir: config['workdir']
sys.path.append(config['workdir'])

cluster = config['cluster']
n_synt_datasets = config['n_synt_datasets']
epsilons = config['epsilons']
MCMC_algorithms = config['MCMC_algorithms']
queries = config['queries']
la_approx = ['LA', 'NoLA']
discretization_levels = ['low', 'high']
algos = config['MCMC_algorithms']

dataset = config['datasets']
dataset_names = [key for key in dataset.keys()]
dataset_files = [value for value in dataset.values()]

original_datasets = config['original_datasets']
original_dataset_names = original_datasets.keys()
original_dataset_files = original_datasets.values()

original_dataset_list, original_query_list, original_epsilon_list = generate_dataset_query_epsilon_products(original_datasets, queries, epsilons)

discretization_datasets = config['discretization_datasets']
discretization_dataset_list, discretization_query_list, discretization_epsilon_list = generate_dataset_query_epsilon_products(discretization_datasets, queries, epsilons)

independence_pruning_datasets = config['independence_pruning_datasets']
independence_pruning_queries = generate_independence_pruning_missing_queries()
independence_dataset_list, independence_query_list, independence_epsilon_list = generate_dataset_query_epsilon_products_independence_queries(independence_pruning_datasets, independence_pruning_queries, epsilons)

linear_regression_dataset_list, linear_regression_epsilon_list = generate_linear_regression_products(epsilons)

singularity: "docker://continuumio/miniconda3:4.12.0"

wildcard_constraints:
    experiment_id="[a-zA-Z\d]{8}"

localrules: csv_results, all

rule all:
    input:
        "results/original_logistic_regression_results.csv",
        "results/synthetic_logistic_regression_results.csv",
        "results/original_classification_results.csv",
        "results/synthetic_classification_results.csv",
        "results/discretization_logistic_regression_results.csv",
        "results/independence_pruning_logistic_regression_results.csv",
        "results/ci_coverage_original_data_results.csv",
        "results/ci_coverage_discretized_data_results.csv",
        "results/ci_coverage_independence_pruning_results.csv",
        "plots/linear_regression_comparison.svg"

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
        dataset_files
    log:
        expand("logs/original_data_generation_dataset_{original_dataset_name}.log",original_dataset_name=dataset_names)
    threads: 1
    resources:
        runtime="30",
        time="00:30:00",
        mem_mb=8000,
        disk_mb=5000,
        partition="short"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_original_datasets.py"


rule generate_parameter_combinations:
    output:
        expand("data/parameter_combinations/napsu_original_model_parameters_{original_dataset_name}_{original_epsilon}e_{original_query_str}.pickle", zip, original_dataset_name=original_dataset_list, original_epsilon=original_epsilon_list, original_query_str=original_query_list),
        expand("data/parameter_combinations/napsu_discretization_model_parameters_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.pickle", zip, discretization_dataset_name=discretization_dataset_list, discretization_epsilon=discretization_epsilon_list, discretization_query_str=discretization_query_list),
        expand("data/parameter_combinations/napsu_independence_pruning_model_parameters_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.pickle", zip, independence_dataset_name=independence_dataset_list, independence_epsilon=independence_epsilon_list, independence_query_str=independence_query_list),
        expand("data/parameter_combinations/napsu_linear_regression_model_parameters_{linear_regression_dataset_name}_{linear_regression_epsilon_str}e.pickle", zip, linear_regression_dataset_name=linear_regression_dataset_list, linear_regression_epsilon_str=linear_regression_epsilon_list)
    log:
        expand("logs/parameter_combinations_original_{original_dataset_name}_{original_epsilon}e_{original_query_str}.log", zip, original_dataset_name=original_dataset_list, original_epsilon=original_epsilon_list, original_query_str=original_query_list),
        expand("logs/parameter_combinations_discretization_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.log", zip, discretization_dataset_name=discretization_dataset_list, discretization_epsilon=discretization_epsilon_list, discretization_query_str=discretization_query_list),
        expand("logs/parameter_combinations_independence_pruning_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.log",  zip, independence_dataset_name=independence_dataset_list, independence_epsilon=independence_epsilon_list, independence_query_str=independence_query_list),
        expand("logs/parameter_combinations_linear_regression_{linear_regression_dataset_name}_{linear_regression_epsilon_str}e.log", zip, linear_regression_dataset_name=linear_regression_dataset_list, linear_regression_epsilon_str=linear_regression_epsilon_list)
    threads: 1
    resources:
        runtime="30",
        time="00:30:00",
        mem_mb=8000,
        disk_mb=5000,
        partition="short"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_parameter_combinations.py"


rule create_original_napsu_models:
    input:
        "data/parameter_combinations/napsu_original_model_parameters_{original_dataset_name}_{original_epsilon}e_{original_query_str}.pickle"
    output:
        "models/napsu_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.dill",
        # timer="napsu_original_model_MCMC_time_vs_epsilon_comparison.csv",
        # experiment_storage="napsu_original_model_experiment_storage_output.csv"
    log:
        "logs/napsu_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.log",
        "logs/inf_data_napsu_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.nc"
    threads: 8
    resources:
        runtime="4320" if cluster == "vorna" else "2880",
        time="72:00:00" if cluster == "vorna" else "48:00:00",
        mem_mb=48000,
        partition="medium"
    # gpu=4
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/create_original_napsu_models.py"


rule generate_synthetic_datasets:
    input:
        "models/napsu_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.dill",
    output:
        "data/synt_datasets/synthetic_dataset_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.pickle"
    log:
        "logs/data_generation_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.log"
    threads: 1
    resources:
        runtime="30",
        time="00:30:00",
        mem_mb=8000,
        disk_mb=5000,
        partition="short"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_synthetic_datasets.py"


rule create_models_for_independence_pruning:
    input:
        "data/parameter_combinations/napsu_independence_pruning_model_parameters_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.pickle",
    output:
        "models/napsu_independence_pruning_model_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.dill",
        #experiment_storage="napsu_independence_pruning_storage.csv",
        #timer="napsu_independence_pruning_timer.csv"
    log:
        "logs/napsu_independence_pruning_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.log",
        "logs/inf_data_independence_pruning_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.nc"
    threads: 8
    resources:
        runtime="4320" if cluster == "vorna" else "2880",
        time="72:00:00" if cluster == "vorna" else "48:00:00",
        mem_mb=48000,
        partition="medium"
    # gpu=4
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/create_models_for_independence_pruning.py"


rule generate_datasets_for_independence_pruning:
    input:
        "models/napsu_independence_pruning_model_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.dill"
    output:
        "data/synt_datasets/synthetic_dataset_independence_pruning_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.pickle"
    log:
        "logs/data_generation_independence_pruning_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.log"
    threads: 1
    resources:
        runtime="30",
        time="00:30:00",
        mem_mb=8000,
        disk_mb=5000,
        partition="short"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_independence_pruning_datasets.py"

rule create_models_for_discretization:
    input:
        "data/parameter_combinations/napsu_discretization_model_parameters_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.pickle",
    output:
        "models/napsu_discretization_model_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.dill",
        #experiment_storage="napsu_discretization_test_storage.csv",
        #timer="napsu_discretization_test_timer.csv"
    log:
        "logs/inf_data_discretization_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.nc",
        "logs/napsu_discretization_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.log"
    threads: 8
    resources:
        runtime="4320" if cluster == "vorna" else "2880",
        time="72:00:00" if cluster == "vorna" else "48:00:00",
        mem_mb=48000,
        partition="medium"
    # gpu=4
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/create_models_with_discretized_data.py"


rule generate_datasets_for_discretized_data:
    input:
        "models/napsu_discretization_model_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.dill"
    output:
        "data/synt_datasets/synthetic_dataset_discretization_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.pickle"
    log:
        "logs/data_generation_discretization_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.log"
    threads: 1
    resources:
        runtime="30",
        time="00:30:00",
        mem_mb=8000,
        disk_mb=5000,
        partition="short"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_datasets_for_discretized_data.py"


rule create_models_for_linear_regression:
    input:
        "data/parameter_combinations/napsu_linear_regression_model_parameters_{linear_regression_dataset_name}_{linear_regression_epsilon_str}e.pickle",
    output:
        "models/napsu_linear_regression_model_{linear_regression_dataset_name}_{linear_regression_epsilon_str}e.dill",
        #experiment_storage="napsu_independence_pruning_storage.csv",
        #timer="napsu_independence_pruning_timer.csv"
    log:
        "logs/napsu_linear_regression_{linear_regression_dataset_name}_{linear_regression_epsilon_str}e.log",
        "logs/inf_data_linear_regression_{linear_regression_dataset_name}_{linear_regression_epsilon_str}e.nc"
    threads: 8
    resources:
        runtime="4320" if cluster == "vorna" else "2880",
        time="72:00:00" if cluster == "vorna" else "48:00:00",
        mem_mb=48000,
        partition="medium"
    # gpu=4
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/create_models_for_linear_regression.py"

rule run_logistic_regression_on_synt:
    input:
        expand("data/synt_datasets/synthetic_dataset_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.pickle", zip, original_dataset_name=original_dataset_list, original_epsilon=original_epsilon_list, original_query_str=original_query_list)
    log:
        "logs/logistic_regression_synthetic_dataset_original_model.log"
    threads: 1
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=32000,
        disk_mb=50000,
        partition="medium"
    output:
        "results/synthetic_logistic_regression_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr_on_synth.py"


rule run_logistic_regression_on_original:
    input:
        original_dataset_files
    output:
        "results/original_logistic_regression_results.csv"
    log:
        expand("logs/logistic_regression_original_dataset_{original_dataset}.log",original_dataset=original_datasets)
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
        expand("data/synt_datasets/synthetic_dataset_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.pickle", zip, original_dataset_name=original_dataset_list, original_epsilon=original_epsilon_list, original_query_str=original_query_list)
    output:
        "results/synthetic_classification_results.csv"
    log:
        "logs/classification_synthetic_dataset_original_models.log"
    threads: 1
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=32000,
        disk_mb=50000,
        partition="medium"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_clf_on_synth.py"


rule run_classification_on_original:
    input:
        original_dataset_files
    output:
        "results/original_classification_results.csv"
    log:
        "logs/classification_original_datasets.log"
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
        expand("data/synt_datasets/synthetic_dataset_discretization_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.pickle", zip, discretization_dataset_name=discretization_dataset_list, discretization_epsilon=discretization_epsilon_list, discretization_query_str=discretization_query_list)
    output:
        "results/discretization_logistic_regression_results.csv"
    log:
        "logs/logistic_regression_discretized_datasets.log"
    threads: 1
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=32000,
        disk_mb=50000,
        partition="medium"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr_on_synth.py"

rule run_logistic_regression_on_independence_pruning_datasets:
    input:
        expand("data/synt_datasets/synthetic_dataset_independence_pruning_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.pickle", zip, independence_dataset_name=independence_dataset_list, independence_epsilon=independence_epsilon_list, independence_query_str=independence_query_list)
    output:
        "results/independence_pruning_logistic_regression_results.csv"
    log:
        "logs/logistic_regression_independence_pruning_datasets.log"
    threads: 1
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=32000,
        disk_mb=50000,
        partition="medium"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr_on_synth.py"


rule run_ci_coverage_on_original_models:
    input:
        "models/napsu_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.dill"
    output:
        "results/ci_coverage_napsu_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.csv"
    log:
        "logs/ci_coverage_napsu_original_models_{original_dataset_name}_{original_epsilon}e_{original_query_str}.log"
    threads: 1
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="medium"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_ci_coverage_on_original_models.py"


rule run_ci_coverage_on_discretized_models:
    input:
        "models/napsu_discretization_model_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.dill"
    output:
        "results/ci_coverage_napsu_discretized_model_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.csv"
    log:
        "logs/ci_coverage_discretization_models_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.log"
    threads: 1
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="medium"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_ci_coverage_on_discretized_models.py"

rule run_ci_coverage_on_independence_pruning_models:
    input:
        "models/napsu_independence_pruning_model_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.dill"
    output:
        "results/ci_coverage_napsu_independence_pruning_model_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.csv"
    log:
        "logs/ci_coverage_independence_pruning_models_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.log"
    threads: 1
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="medium"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_ci_coverage_on_independence_pruning_models.py"


rule combine_ci_results_from_original_models:
    input:
        expand("results/ci_coverage_napsu_original_model_{original_dataset_name}_{original_epsilon}e_{original_query_str}.csv",zip,original_dataset_name=original_dataset_list,original_epsilon=original_epsilon_list,original_query_str=original_query_list)
    output:
        "results/ci_coverage_original_data_results.csv"
    log:
        "logs/combine_ci_results_from_original_models.log"
    threads: 1
    resources:
        runtime="30",
        time="00:30:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/combine_csvs.py"

rule combine_ci_results_from_discretized_models:
    input:
        expand("results/ci_coverage_napsu_discretized_model_{discretization_dataset_name}_{discretization_epsilon}e_{discretization_query_str}.csv", zip, discretization_dataset_name=discretization_dataset_list, discretization_epsilon=discretization_epsilon_list, discretization_query_str=discretization_query_list)
    output:
        "results/ci_coverage_discretized_data_results.csv"
    log:
        "logs/combine_ci_results_from_discretized_model.log"
    threads: 1
    resources:
        runtime="30",
        time="00:30:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/combine_csvs.py"


rule combine_ci_results_from_independence_pruning_models:
    input:
        expand("results/ci_coverage_napsu_independence_pruning_model_{independence_dataset_name}_{independence_epsilon}e_{independence_query_str}.csv", zip, independence_dataset_name=independence_dataset_list, independence_epsilon=independence_epsilon_list, independence_query_str=independence_query_list)
    output:
        "results/ci_coverage_independence_pruning_results.csv"
    log:
        "logs/combine_ci_results_from_independence_pruning_model.log"
    threads: 1
    resources:
        runtime="30",
        time="00:30:00",
        mem_mb=16000,
        disk_mb=50000,
        partition="short"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/combine_csvs.py"

rule compare_original_lr_results:
    input:
        "results/synthetic_logistic_regression_results.csv",
        "results/original_logistic_regression_results.csv"
    output:
        report(expand("plots/lr_comparison_{original_dataset}.svg",original_dataset=original_dataset_names))
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
        report(expand("plots/clf_comparison_{original_dataset}.svg",original_dataset=original_dataset_names))
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

rule create_linear_regression_plots:
    input:
        expand("models/napsu_linear_regression_model_{linear_regression_dataset_name}_{linear_regression_epsilon_str}e.dill", zip, linear_regression_dataset_name=linear_regression_dataset_list, linear_regression_epsilon_str=linear_regression_epsilon_list)
    output:
        report("plots/linear_regression_comparison.svg")
    log:
        "logs/linear_regression_plot.log"
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
        "scripts/plot_linear_regression_results.py"

rule clean_slurm_logs:
    shell:
        "rm slurm_logs/*"
