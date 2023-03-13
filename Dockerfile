FROM condaforge/mambaforge:latest
LABEL io.github.snakemake.containerized="true"
LABEL io.github.snakemake.conda_env_hash="18ac06b578cdad7831057ad474c1ae7dfa2365a90ada24f1b8a91cf60657200d"

# Step 1: Retrieve conda environments

# Conda environment:
#   source: envs/analysis.yaml
#   prefix: /conda-envs/90520af2200ba5c8cd2b335f5d35935d
#   name: napsu-analysis-env
#   channels:
#     - conda-forge
#     - bioconda
#   dependencies:
#     - python=3.8.10
#     - snakemake=7.24.1
#     - scikit-learn=1.2.2
#     - pandas=1.5.3
#     - xgboost=1.7.4
#     - lightgbm=3.3.5
#     - seaborn=0.12.2
#   #variables:
#   #  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wrk-vakka/users/jarlehti/miniconda3/lib
RUN mkdir -p /conda-envs/90520af2200ba5c8cd2b335f5d35935d
COPY envs/analysis.yaml /conda-envs/90520af2200ba5c8cd2b335f5d35935d/environment.yaml

# Conda environment:
#   source: envs/napsu.yaml
#   prefix: /conda-envs/32be39d07b4271fb3f64aa9cb1955485
#   name: napsu-model-env
#   channels:
#     - conda-forge
#     - anaconda
#     - bioconda
#   dependencies:
#     - python=3.8.10
#     - snakemake=7.24.1
#     - jax=0.4.6
#     - pandas=1.5.3
#     - arviz=0.15.1
#     - dill=0.3.6
#     - python-graphviz=0.20.1
#     - pip
#     - pip:
#           - git+https://github.com/DPBayes/d3p.git@master#egg=d3p@0.2.0
#           - git+https://github.com/ryan112358/private-pgm.git@ee90ac4ea5901654a0e86f0be4e217ddcf5b53bd
#           - jaxopt=0.6
#   #variables:
#   #  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wrk-vakka/users/jarlehti/miniconda3/lib
RUN mkdir -p /conda-envs/32be39d07b4271fb3f64aa9cb1955485
COPY envs/napsu.yaml /conda-envs/32be39d07b4271fb3f64aa9cb1955485/environment.yaml

# Step 2: Generate conda environments

RUN mamba env create --prefix /conda-envs/90520af2200ba5c8cd2b335f5d35935d --file /conda-envs/90520af2200ba5c8cd2b335f5d35935d/environment.yaml && \
    mamba env create --prefix /conda-envs/32be39d07b4271fb3f64aa9cb1955485 --file /conda-envs/32be39d07b4271fb3f64aa9cb1955485/environment.yaml && \
    mamba clean --all -y
