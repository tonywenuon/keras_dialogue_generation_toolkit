

data_set='example_data'

python train_transformer.py --data_set=$data_set \
       --exp_name=transformer \
       --batch_size=40 \
       --src_seq_length=30 \
       --tar_seq_length=30 \
       --early_stop_patience=2 \
       --lr_decay_patience=1 \
       --lr=0.001




