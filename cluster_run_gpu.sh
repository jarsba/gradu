source ~/.bashrc
conda activate py38
module load Singularity/3.11.0-GCC-12.2.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

snakemake --use-conda --use-singularity --singularity-args '\\-\\-nv' --profile `pwd`/workflow --directory "/wrk-vakka/users/jarlehti/gradu" --touch
snakemake --use-conda --use-singularity --singularity-args '\\-\\-nv' --profile `pwd`/workflow --directory "/wrk-vakka/users/jarlehti/gradu" --until RULE

