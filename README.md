## The proposed FedGDF algorithm is implemented under the name FedKD in our code.

#Local Cora
python run_node_task_main.py --dataset Cora --num_workers 20 --overlapping_rate 0.0 --is_iid iid --alg_method Local --gpu_id 0

#FedGDF Cora
python run_node_task_main.py --dataset Cora --num_workers 20 --overlapping_rate 0.0 --is_iid iid --alg_method FedKD --ALPHA 1.0 --lam_real 0 --sample_num 100 --lam2 0 --gpu_id 0

#FedTAD Cora
python run_node_task_main.py --dataset Cora --num_workers 20 --overlapping_rate 0.0 --is_iid iid --alg_method FedTAD --ALPHA 1.0  --sample_num 100 --lam2 1 --topk 5 --gpu_id 0

#FedProto
python run_node_task_main.py --dataset Cora --num_workers 20 --overlapping_rate 0.0 --is_iid iid --alg_method Fedproto --w_proto 0.5 --gpu_id 0

#FGPL
python run_node_task_main.py --dataset Cora --num_workers 20 --overlapping_rate 0.0 --is_iid iid --alg_method FGPL --gpu_id 0

#FedTGP
python run_node_task_main.py --dataset Cora --num_workers 20 --overlapping_rate 0.0 --is_iid iid --alg_method FedTGP --w_proto 0.5 --gpu_id 0