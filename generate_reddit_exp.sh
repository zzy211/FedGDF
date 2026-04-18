#!/bin/bash

# Define parameters
dataset="Reddit"
alg_methods="FedProto"  # Modify this to your desired algorithm, e.g., "FedKD" or "FedAvg"
gpu_id=3
run_all_dir="script"


for num_clients in 10 20 50; do
    
    base_dir="script/${num_clients}clients"
    dataset_dir="$base_dir/$dataset"
    mkdir -p "$dataset_dir"
    
    script_name="${alg_methods,,}_${dataset,,}_${num_clients}.sh"
    script_path="$dataset_dir/$script_name"
    
    cat > "$script_path" << SCRIPT
#!/bin/bash
# python run_node_task_main.py --dataset "$dataset" --num_workers "$num_clients" --overlapping_rate 0.0 --is_iid iid --alg_method Local --gpu_id "$gpu_id"
python run_node_task_main.py --dataset "$dataset" --num_workers "$num_clients" --overlapping_rate 0.0 --is_iid iid --alg_method Fedproto --gpu_id $gpu_id --w_proto 0.5
# python run_node_task_main.py --dataset "$dataset" --num_workers "$num_clients" --overlapping_rate 0.0 --is_iid iid --alg_method FedTAD --gpu_id $gpu_id  --ALPHA 1.0  --sample_num 100 --lam2 1 --topk 5
# python run_node_task_main.py --dataset "$dataset" --num_workers "$num_clients" --overlapping_rate 0.0 --is_iid iid --alg_method FedTGP  --gpu_id $gpu_id --w_proto 0.5
# python run_node_task_main.py --dataset "$dataset" --num_workers "$num_clients" --overlapping_rate 0.0 --is_iid iid --alg_method FedKD  --ALPHA 1.0 --lam_real 0 --sample_num 100 --lam2 0 --gpu_id $gpu_id
# python run_node_task_main.py --dataset "$dataset" --num_workers "$num_clients" --overlapping_rate 0.0 --is_iid iid --alg_method FGPL --gpu_id $gpu_id
SCRIPT

    chmod +x "$script_path"
    
    echo "nohup sh $script_path > $dataset_dir/${script_name%.sh}.log 2>&1 &" >> "$run_all_dir/run_all.sh"
    
    echo "Created script: $script_path"

done

chmod +x "$run_all_dir/run_all.sh"

echo "Completed!"
echo "All scripts have been generated in: $run_all_dir"
echo "To run all experiments, execute: sh $run_all_dir/run_all.sh"