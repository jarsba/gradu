import numpyro


def get_MCMC_algorithm(name):
    if name == 'NUTS':
        return numpyro.infer.NUTS
    elif name == 'HMC':
        return numpyro.infer.HMC
    else:
        raise ValueError(f"{name} not recognized MCMC algorithm")


def get_dataset(name):
    if name == 'adult':
        return "data/datasets/cleaned_adult_data_v2.csv"
    elif name == 'dummy':
        return "data/datasets/binary_data4d.csv"
    else:
        raise ValueError(f"{name} not recognized dataset")