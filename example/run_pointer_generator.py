#!/bin/bash


cd '../example'
python_exe='train_pointer_generator.py'

       #--mode='decode' \
       #--single_pass=True \
       #--mode='train' \
       #--single_pass=False \
python $python_exe \
       --mode='train' \
       --single_pass=False \
       --data_set='example_data' \
       --exp_name='pointer_generator_with_coverage' \
       --batch_size=50 \
       --hidden_dim=200 \
       --emb_dim=200 \
       --max_enc_steps=30 \
       --max_dec_steps=30 \
       --beam_size=5 \
       --min_dec_steps=10 \
       --vocab_size=50000 \
       --outputs_dir='outputs' \
       --log_root='log' \
       --lr=0.01 \
       --pointer_gen=True \
       --coverage=True \
       --cov_loss_wt=1.0 \



