# NNsurv (package in progress)

Neural networks based on a discrete-time model to predict the survival duration

## Introduction
Three versions of Neural networks based on a discrete-time model to predict the survival duration are available in this package:
  - NNsurv (version 1): basic structure as proposed by biganzoli *et al* (adding cross-validation procedure) 
  - NNsurv deep (version 2): deep structure of NNsurv (not available for now) 
  - NNsurvK (version 3): NNsurv with multivariate outputs (and adding of Fused-Lasso regularization and Kaplan-Meier censoring indicator)

## Running
python NNsurv.py "data/simuZ_vf_KIRC.csv" "data/surv_time_vf_KIRC.csv" "data/right_cens_vf_KIRC.csv" 0

python NNsurvK.py "data/simuZ_vf_KIRC.csv" "data/surv_time_vf_KIRC.csv" "data/right_cens_vf_KIRC.csv" 0
