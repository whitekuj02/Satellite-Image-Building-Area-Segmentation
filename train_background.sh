#!/bin/bash

export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH

. ~/anaconda3/etc/profile.d/conda.sh

conda activate

conda activate aicon

today=`date +%y%m%d_%H%M%S`

echo $today

nohup python3 -u train.py -b 32 -lr 2e-04 -aug -aug_fr -aug_bc -aug_cl -nep 20 -pf efun7_fr_bc_cl_$today >log/log.$today&
