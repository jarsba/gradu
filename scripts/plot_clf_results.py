import seaborn as sns
import pandas as pd

original_results_csv = snakemake.input[0]
synthetic_results_csv = snakemake.input[0]

original_data = pd.read_csv(original_results_csv)
synthetic_data = pd.read_csv(synthetic_results_csv)

n_original_datasets = original_data['dataset_name'].nunique()
dataset_names = original_data['dataset_name'].unique()

for dataset_name in dataset_names:
    synthetic_results_with_dataset = synthetic_data[synthetic_data['dataset_name'] == dataset_name]

    n_index = synthetic_results_with_dataset['dataset_index'].nunique()
    for index in range(n_index):
        synthetic_results_with_dataset_index = synthetic_results_with_dataset[
            synthetic_results_with_dataset['dataset_index'] == index]

        original_data_with_dataset = original_data[original_data['dataset_name'] == dataset_name]
        original_data_with_dataset['MCMC_algorithm'] = 'Original'
        original_data_with_dataset['epsilon'] = 'Inf'
        combined_data = pd.concat(synthetic_results_with_dataset_index, original_data_with_dataset)

        g = sns.FacetGrid(combined_data, col="MCMC_algorithm", hue="model_name")
        g.map(sns.barplot, "epsilon", "accuracy")
        g.savefig(f"plots/clf_comparison_{index}.svg")