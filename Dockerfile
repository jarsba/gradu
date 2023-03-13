FROM condaforge/mambaforge:latest
LABEL io.github.snakemake.containerized="true"
LABEL io.github.snakemake.conda_env_hash="37c28ebe21e7da5a42649adaa507cd50119292ab3c2151ce001a103defbd4278"

# Step 1: Retrieve conda environments

# Conda environment:
#   source: envs/analysis.yaml
#   prefix: /conda-envs/2487cf1f47f6a6827bc15a426defa90b
#   name: napsu-analysis-env
#   channels:
#     - conda-forge
#     - bioconda
#   dependencies:
#     - python=3.8.10
#     - snakemake=7.16.0
#     - scikit-learn=1.1.2
#     - pandas=1.5.0
#     - xgboost=1.6.2
#     - lightgbm=3.3.2
#     - seaborn=0.12.0
#   #variables:
#   #  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wrk-vakka/users/jarlehti/miniconda3/lib
RUN mkdir -p /conda-envs/2487cf1f47f6a6827bc15a426defa90b
COPY envs/analysis.yaml /conda-envs/2487cf1f47f6a6827bc15a426defa90b/environment.yaml

# Conda environment:
#   source: envs/napsu.yaml
#   prefix: /conda-envs/f0c304fa62b70b2f288dbde450550689
#   name: napsu-model-env
#   channels:
#     - conda-forge
#     - anaconda
#     - bioconda
#   dependencies:
#     - python=3.8.10
#     - snakemake=7.16.0
#     - jax=0.3.17
#     - pandas=1.5.0
#     - arviz=0.12.1
#     - dill=0.3.5.1
#     - python-graphviz=0.20.1
#     - pip
#     - pip:
#           - git+https://github.com/DPBayes/d3p.git@master#egg=d3p@0.2.0
#           - git+https://github.com/ryan112358/private-pgm.git@ee90ac4ea5901654a0e86f0be4e217ddcf5b53bd
#           - jaxopt=0.6
#   #variables:
#   #  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wrk-vakka/users/jarlehti/miniconda3/lib
RUN mkdir -p /conda-envs/f0c304fa62b70b2f288dbde450550689
COPY envs/napsu.yaml /conda-envs/f0c304fa62b70b2f288dbde450550689/environment.yaml

# Step 2: Generate conda environments

RUN mamba env create --prefix /conda-envs/2487cf1f47f6a6827bc15a426defa90b --file /conda-envs/2487cf1f47f6a6827bc15a426defa90b/environment.yaml && \
    mamba env create --prefix /conda-envs/f0c304fa62b70b2f288dbde450550689 --file /conda-envs/f0c304fa62b70b2f288dbde450550689/environment.yaml && \
    mamba clean --all -y
