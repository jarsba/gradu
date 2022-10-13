workdir: "/home/local/jarlehti/projects/gradu"
configfile: "config.yaml"

n_synt_datasets = config['n_synt_datasets']
dataset = config['datasets']
dataset_names = [key for key in dataset.keys()]
dataset_files = [value for value in dataset.values()]
epsilons = config['epsilons']
MCMC_algorithms = config['MCMC_algorithms']
dataset_index = [i for i in range(n_synt_datasets)]
queries = config['queries']

def query_dataset_product(datasets, queries):
    query_dataset_product = []

    for dataset in datasets:
        dataset_queries = queries[dataset]
        for query in dataset_queries:
            if query == []:
                query_str = "empty"
            else:
                query_str = "".join(query)
            dataset_query = f"{dataset}_{query_str}"
            query_dataset_product.append(dataset_query)

    return query_dataset_product

queries_dataset_product  = query_dataset_product(dataset_names, queries)

rule all:
    input:
        expand("plots/lr_comparison_{dataset}.svg", dataset=dataset_names),
        expand("plots/clf_comparison_{dataset}.svg", dataset=dataset_names)


rule run_napsu:
    input:
        expand("data/datasets/{dataset}", dataset=dataset_files)
    output:
        expand("models/napsu_{dataset_query}_{epsilon}e_{MCMC_algorithm}.dill", dataset_query=queries_dataset_product, epsilon=epsilons, MCMC_algorithm=MCMC_algorithms)
    #log:
    #    expand("logs/napsu/napsu_{dataset}_{epsilon}e_{MCMC_algorithm}.log", dataset=dataset_names, epsilon=epsilons, MCMC_algorithm=MCMC_algorithms)
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/run_napsu.py"


rule generate_synt_datasets:
    input:
        expand("models/napsu_{dataset_query}_{epsilon}e_{MCMC_algorithm}.dill", dataset_query=queries_dataset_product, epsilon=epsilons, MCMC_algorithm=MCMC_algorithms,)
    output:
        expand("data/synt_datasets/synthetic_dataset_{i}_{dataset_query}_{epsilon}e_{MCMC_algorithm}.csv", i=dataset_index, dataset_query=queries_dataset_product, epsilon=epsilons, MCMC_algorithm=MCMC_algorithms)
    conda:
        "envs/napsu.yaml"
    script:
        "scripts/generate_datasets.py"


rule run_logistic_regression_on_synt:
    input:
        expand("data/synt_datasets/synthetic_dataset_{i}_{dataset_query}_{epsilon}e_{MCMC_algorithm}.csv", i=dataset_index, dataset_query=queries_dataset_product, epsilon=epsilons, MCMC_algorithm=MCMC_algorithms)
    output:
        "results/synthetic_logistic_regression_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr.py"


rule run_logistic_regression_on_original:
    input:
        expand("data/datasets/{dataset}", dataset=dataset_files)
    output:
        "results/original_logistic_regression_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_lr.py"


rule run_classification_on_synt:
    input:
        expand("data/synt_datasets/synthetic_dataset_{i}_{dataset_query}_{epsilon}e_{MCMC_algorithm}_{query}.csv", i=dataset_index, dataset_query=queries_dataset_product, epsilon=epsilons, MCMC_algorithm=MCMC_algorithms)
    output:
        "results/synthetic_classification_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_clf.py"


rule run_classification_on_original:
    input:
        expand("data/datasets/{dataset}", dataset=dataset_files)
    output:
        "results/original_classification_results.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/run_clf.py"


#rule compare_datasets:
#    input:
#        expand("data/datasets/{dataset}", dataset=dataset_files),
#        expand("data/synt_datasets/synthetic_dataset_{i}_{dataset}_{epsilon}e_{MCMC_algorithm}.csv", i=dataset_index, dataset=dataset_names, epsilon=epsilons, MCMC_algorithm=MCMC_algorithms)
#    output:
#        "results/comparison_{dataset}_vs_{i}_{parameters}.csv"
#    conda:
#        "envs/analysis.yaml"
#    script:
#        "scripts/compare_datasets.py"


rule compare_lr_results:
    input:
        "results/synthetic_logistic_regression_results.csv",
        "results/original_logistic_regression_results.csv",
    output:
        report(expand("plots/lr_comparison_{dataset}.svg", dataset=dataset_names))
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/plot_results.py"


rule compare_clf_results:
    input:
        "results/synthetic_classification_results.csv",
        "results/original_classification_results.csv"
    output:
        report(expand("plots/clf_comparison_{dataset}.svg", dataset=dataset_names))
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/plot_results.py"
