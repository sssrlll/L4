A=woz.en
B=cnn_dailymail
C=wikisql

bash train.sh --device_ids 0 --lamb 0.2 --train_batch_size 4 --n_train_epochs 9 --seq_train_type lll --model_name gpt2 --add_task_tokens --seed 531 --gen_lm_sample_percentage 0.2 --distil --mutual_distil --tasks $A $B $C 

wait

bash test.sh --device_ids 0 --lamb 0.2 --test_batch_size 5 --n_train_epochs 9 --seq_train_type lll --model_name gpt2 --add_task_tokens --seed 531 --gen_lm_sample_percentage 0.2 --distil --mutual_distil --tasks $A $B $C 
