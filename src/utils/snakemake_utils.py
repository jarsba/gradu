from src.utils.keygen import get_key, get_hash


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


def generate_products(queries_dataset_product, epsilons, MCMC_algorithms):
    products = []
    for query_dataset in queries_dataset_product:
        for epsilon in epsilons:
            for algo in MCMC_algorithms:
                # id = get_hash(query_dataset, epsilon, algo)
                # product_str = f"{id}_{query_dataset}_{epsilon}e_{algo}"
                product_str = f"{query_dataset}_{epsilon}e_{algo}"
                products.append(product_str)

    return products
