source ~/.bashrc
conda activate py38
module load Singularity/3.11.0-GCC-12.2.0

snakemake --use-conda -j 60 --use-singularity --profile `pwd`/snakemake-slurm --directory "/wrk-vakka/users/jarlehti/gradu" --touch
snakemake --use-conda -j 60 --use-singularity --profile `pwd`/snakemake-slurm --directory "/wrk-vakka/users/jarlehti/gradu" --until RULE

