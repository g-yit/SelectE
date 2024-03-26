#! /bin/bash
python_env=F:/ProgramData/Anaconda3/envs/pykeen/python

## WN18RR  0.493	0.452	0.509	0.574
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "WN18RR" --seed 1234 --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 20 --min_lr 0.00001 --batch_size 1500 --log_epoch 2 --neg_ratio 1 --input_drop 0.2 --hidden_drop 0.5 --feature_map_drop 0.2 --opt "Adam" --learning_rate 0.001 --weight_decay 5e-4 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 300 --momentum 0.9 --save_name "./model/wn18rr.pt"
## KINSHIP 0.882	0.816	0.942	0.985
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "KINSHIP" --seed 2024 --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 20 --min_lr 0.00001 --batch_size 800 --log_epoch 2 --neg_ratio 1 --input_drop 0.3 --hidden_drop 0.1 --feature_map_drop 0.4 --opt "Adam" --learning_rate 0.001 --weight_decay 5e-3 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 220 --momentum 0.9 --save_name "./model/KINSHIP.pt"
## UMLS    0.923	0.852	0.992	0.999
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "UMLS" --seed 2022 --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 24 --min_lr 0.00001 --batch_size 896 --log_epoch 2 --neg_ratio 1 --input_drop 0.2 --hidden_drop 0.1 --feature_map_drop 0.3 --opt "Adam" --learning_rate 0.001 --weight_decay 5e-3 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 400 --momentum 0.9 --save_name "./model/UMLS.pt"
## FB15K-237 0.353	0.257	0.390	0.544
##$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "FB15k-237" --seed 2022 --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 32 --min_lr 0.000005 --batch_size 2000 --log_epoch 2 --neg_ratio 1 --input_drop 0.2 --hidden_drop 0.4 --feature_map_drop 0.3 --opt "Adam" --learning_rate 0.0005 --weight_decay 5e-4 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 800 --momentum 0.9 --save_name "./model/FB15K237.pt"
## WN18 0.951 0.947 0.953 0.957
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "WN18" --seed 2022 --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 20 --min_lr 0.00001 --batch_size 1500 --log_epoch 2 --neg_ratio 1 --input_drop 0.3 --hidden_drop 0.5 --feature_map_drop 0.1 --opt "Adam" --learning_rate 0.0003 --weight_decay 5e-8 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 1000 --momentum 0.9 --save_name "./model/WN18.pt"
## yago 0.563	0.490	0.609	0.697
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "YAGO3-10" --seed 2022--embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 20 --min_lr 0.00001 --batch_size 1500 --log_epoch 2 --neg_ratio 1 --input_drop 0.2 --hidden_drop 0.1 --feature_map_drop 0.3 --opt "Adam" --learning_rate 0.001 --weight_decay 5e-9 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 350 --momentum 0.9 --save_name "./model/yago.pt"
## FB15K 0.812	0.760	0.849	0.898
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "FB15k" --seed 2022--embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 20 --min_lr 0.00001 --batch_size 1500 --log_epoch 2 --neg_ratio 1 --input_drop 0.2 --hidden_drop 0.1 --feature_map_drop 0.3 --opt "Adam" --learning_rate 0.001 --weight_decay 5e-8 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 500 --momentum 0.9 --save_name "./model/FB15K.pt"