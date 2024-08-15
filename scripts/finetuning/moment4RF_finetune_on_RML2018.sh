#!/bin/bash

############################################
## Fine-tuning mode: Linear probing
############################################

 ### dataset:RML-2018a
 python scripts/finetuning/forecasting.py\
  --finetuning_mode 'linear-probing'\
  --config 'configs/forecasting/nhits.yaml'\
  --gpu_id 3\
  --forecast_horizon 720\
  --init_lr 0.0001\
  --dataset_names '/user_home/WirelessData/RML2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
