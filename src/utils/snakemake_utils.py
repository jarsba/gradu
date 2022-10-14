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
