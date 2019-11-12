#!/bin/bash
#####################################################################
# Configuracao                                                      #
#####################################################################

# Ativa o AMBER
source ~/software/amber18/amber.sh

# Numero de processadores
NSLOTS=8

# Nomes dos inputs iniciais.
init="ionized"     # Nome da topologia (system.prmtop)
prev="ionized"     # Nome inicial para os arquivos (system.rst7, system.prmtop ...)



#####################################################################
# Funcoes                                                           #
#####################################################################
# Função para rodar o sander
run_pmemd() {
mpirun -n ${NSLOTS} sander.MPI -O \
  -i amber_inputs/${input}.in \
  -o ${run}.out \
  -r ${run}.rst7 \
  -x ${run}.nc \
-inf ${run}.mdinfo \
  -p ${init}.prmtop \
  -c ${prev}.rst7 \
-ref ${prev}.rst7 
}


run_pmemd_cuda() {
pmemd.cuda -O \
  -i amber_inputs/${input}.in \
  -o ${run}.out \
  -r ${run}.rst7 \
  -x ${run}.nc \
-inf ${run}.mdinfo \
  -p ${init}.prmtop \
  -c ${prev}.rst7 \
-ref ${prev}.rst7
}



#--------------------------------------------------------------------
# Programa
#--------------------------------------------------------------------

# Step 1 - Minimization ---------------------------------------------
for run in "min_fix" "min_lib" ; do
    input=${run}
#    run_pmemd 
    prev=${run}
done

# Step 2 - Heat --------------------------------------------
for run in "heat" "density" ; do
    input=${run}
#    run_pmemd
    prev=${run}
done

# Step 3 - Equilibration --------------------------------------------
for run in "equil_fix" "equil_lib" ; do
    input=${run}
#    run_pmemd_cuda
    prev=${run}
done


# Step 4 - Production -----------------------------------------------
# A produção acontece em 10 partes
input="prod"   # Na producao o "input" é fixo, e se chama prod
for part in $(seq 1 10) ; do
    run=prod_${part}
#    run_pmemd_cuda
    prev=${run}
done


# Step 5 - CONTINUE -----------------------------------------------
# A produção acontece em 10 partes
input="prod"   # Na producao o "input" é fixo, e se chama prod
for part in $(seq 11 100) ; do
    run=prod_${part}
    run_pmemd_cuda
    prev=${run}
done
