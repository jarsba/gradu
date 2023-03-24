import sys

configfile: "config.yaml"
workdir: config['workdir']
sys.path.append(config['workdir'])

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
original_dataset_names = [key for key in original_datasets.keys()]
original_dataset_files = [value for value in original_datasets.values()]

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
        dataset_files
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


rule generate_parameter_combinations:
    output:
        "data/parameter_combinations/napsu_original_model_parameters_{dataset_name}_{epsilon}e_{query_str}.pickle",
        "data/parameter_combinations/napsu_discretization_model_parameters_{dataset_name}_{epsilon}e_{query_str}",
        "data/parameter_combinations/napsu_independence_pruning_model_parameters_{dataset_name}_{epsilon}e_{query_str}.pickle",
    log:
        "logs/parameter_combinations_{dataset_name}_{epsilon}e_{query_str}.log"
    threads: 1
    resources:
        runtime="10",
        time="00:10:00",
        mem_mb=2000,
        disk_mb=5000,
        partition="short"
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_parameter_combinations.py"

rule create_original_napsu_models:
    input:
        "data/parameter_combinations/napsu_original_model_parameters_{dataset_name}_{epsilon}e_{query_str}.pickle"
    output:
        model_name="models/napsu_original_model_{dataset_name}_{epsilon}e_{query_str}.dill",
        #timer="napsu_original_model_MCMC_time_vs_epsilon_comparison.csv",
        #experiment_storage="napsu_original_model_experiment_storage_output.csv"
    log:
        "logs/napsu_original_model_{dataset_name}_{epsilon}e_{query_str}.log",
        "logs/inf_data_napsu_original_model_{dataset_name}_{epsilon}e_{query_str}.nc"
    threads: 8
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=48000,
        partition="medium"
    # gpu=4
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/create_original_napsu_models.py"


rule generate_synthetic_datasets:
    input:
        "models/napsu_original_model_{dataset_name}_{epsilon}e_{query_str}.dill",
    output:
        "data/synt_datasets/synthetic_dataset_original_model_{dataset_name}_{epsilon}e_{query_str}.pickle"
    log:
        "logs/data_generation_original_model_{dataset_name}_{epsilon}e_{query_str}.log"
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
        "data/parameter_combinations/napsu_independence_pruning_model_parameters_{dataset_name}_{epsilon}e_{query_str}.pickle",
    output:
        model_name="models/napsu_independence_pruning_model_{dataset_name}_{epsilon}e_{query_str}.dill",
        #experiment_storage="napsu_independence_pruning_storage.csv",
        #timer="napsu_independence_pruning_timer.csv"
    log:
        "logs/napsu_independence_pruning__{dataset_name}_{epsilon}e_{query_str}.log",
        "logs/inf_data_independence_pruning_{dataset_name}_{epsilon}e_{query_str}.nc"
    threads: 8
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=48000,
        partition="medium"
    # gpu=4
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/create_models_for_independence_pruning.py"


rule generate_datasets_for_independence_pruning:
    input:
        "models/napsu_independence_pruning_model_{dataset_name}_{epsilon}e_{query_str}.dill"
    output:
        "data/synt_datasets/synthetic_dataset_independence_pruning_{dataset_name}_{epsilon}e_{query_str}.pickle"
    log:
        "logs/data_generation_independence_pruning_{dataset_name}_{epsilon}e_{query_str}.log"
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
        "data/parameter_combinations/napsu_discretization_model_parameters_{dataset_name}_{epsilon}e_{query_str}",
    output:
        model_name="models/napsu_discretization_model_{dataset_name}_{epsilon}e_{query_str}.dill",
        #experiment_storage="napsu_discretization_test_storage.csv",
        #timer="napsu_discretization_test_timer.csv"
    log:
        "logs/inf_data_discretization_{dataset_name}_{epsilon}e_{query_str}.nc",
        "logs/napsu_discretization_{dataset_name}_{epsilon}e_{query_str}.log"
    threads: 8
    resources:
        runtime="2880",
        time="48:00:00",
        mem_mb=48000,
        partition="medium"
    # gpu=4
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/create_models_with_discretized_data.py"


rule generate_datasets_for_discretized_data:
    input:
        "models/napsu_discretization_model_{dataset_name}_{epsilon}e_{query_str}.dill"
    output:
        "data/synt_datasets/synthetic_dataset_discretization_{dataset_name}_{epsilon}e_{query_str}.pickle"
    log:
        "logs/data_generation_discretization_{dataset_name}_{epsilon}e_{query_str}.log"
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
        glob_wildcards("data/synt_datasets/synthetic_dataset_original_model_{experiment_product}.pickle")
    log:
        "logs/logistic_regression_synthetic_dataset_original_model.log"
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
        original_dataset_files
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
        glob_wildcards("data/synt_datasets/synthetic_dataset_original_model_{experiment_product}.pickle")
    output:
        "results/synthetic_classification_results.csv"
    log:
        "logs/classification_synthetic_dataset_original_models.log"
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
        glob_wildcards("data/synt_datasets/synthetic_dataset_discretization_{discretization_product}.pickle")
    output:
        "results/discretization_logistic_regression_results.csv"
    log:
        "logs/logistic_regression_discretized_datasets.log"
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
        glob_wildcards("data/synt_datasets/synthetic_dataset_independence_pruning_{{independence_pruning_product}}.pickle")
    output:
        "results/independence_pruning_logistic_regression_results.csv"
    log:
        "logs/logistic_regression_independence_pruning_datasets.log"
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
        glob_wildcards("models/napsu_original_model_{experiment_product}.dill")
    output:
        "results/ci_coverage_original_data_results.csv"
    log:
        "logs/ci_coverage_napsu_original_modelss.log"
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
        glob_wildcards("models/napsu_discretization_model_{discretization_product}.dill")
    output:
        "results/ci_coverage_discretized_data_results.csv"
    log:
        "logs/ci_coverage_discretization_models.log"
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
        glob_wildcards("models/napsu_independence_pruning_model_{independence_pruning_product}.dill")
    output:
        "results/ci_coverage_independence_pruning_results.csv"
    log:
        "logs/ci_coverage_independence_pruning_models.log"
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
