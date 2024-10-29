#!/bin/bash

#SBATCH -N 4                # Demande 2 nœuds
#SBATCH -n 320              # Demande 160 tâches au total (80 par nœud)
#SBATCH --gres=gpu:2        # 2 GPU
#SBATCH -p small            # Utilise la partition 'small'
#SBATCH --ntasks-per-node=80 # Demande 80 tâches par nœud
#SBATCH --time=02:00:00     # Définit une limite de temps de 2 heures

      

module purge

module load conda/22.11.1

      

# Preparation de l'environnement d'execution

myProjectDir="/work/${GROUPE}/${USER}/trocr_handwritten"

myExec="scripts/launch_layout.py"

myWorkDir="/tmpdir/${USER}/${SLURM_JOBID}"

mkdir -p "${myWorkDir}"

cd "${myWorkDir}"

cp "${0}" .

cp ${myProjectDir}/${myExec} .  

# Exécution

source ${myProjectDir}/.venv/bin/activate

python ${myProjectDir}/${myExec}

deactivate

sleep 10

jobinfo ${SLURM_JOBID}