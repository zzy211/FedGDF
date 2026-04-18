#!/bin/bash

# Define parameters
datasets=("Cora" "Citeseer" "Pubmed" "Cs" "Physics")
alg_methods="FGPL"
server_id=4
gpu_id=4

# Iterate through all datasets and client counts
for num_clients in 50; do
    for dataset in "${datasets[@]}"; do
        # Create base directories and modify paths based on client count
        base_dir="script/${num_clients}clients"  # Set directories according to the number of clients
        dataset_dir="$base_dir/$dataset"
        
        # Create dataset directories
        mkdir -p "$dataset_dir"
        
        # Script file name
        script_name="${alg_methods,,}_${dataset,,}_${num_clients}.sh"
        script_path="$dataset_dir/$script_name"
        
        # Write script content
        cat > "$script_path" << SCRIPT
#!/bin/bash
# python run_node_task_main.py --dataset $dataset --num_workers $num_clients --overlapping_rate 0.0 --is_iid iid --alg_method Local --gpu_id $gpu_id
# python run_node_task_main.py --dataset $dataset --num_workers $num_clients --overlapping_rate 0.0 --is_iid iid --alg_method FedTAD --gpu_id $gpu_id  --ALPHA 1.0  --sample_num 100 --lam2 1 --topk 5
python run_node_task_main.py --dataset $dataset --num_workers $num_clients --overlapping_rate 0.0 --is_iid iid --alg_method FGPL --gpu_id $gpu_id
SCRIPT
        
        # Make script executable
        chmod +x "$script_path"
        
        # Generate run command and add to overall run script
        echo "nohup sh $script_path > $dataset_dir/${script_name%.sh}.log 2>&1 &" >> "$base_dir/run_all.sh"
        
        echo "Creating script: $script_path"
    done
done

# Make the main run script executable
chmod +x "$base_dir/run_all.sh"

echo "Completed!"
echo "All scripts have been generated in: $base_dir"
echo "To run all experiments, execute: sh $base_dir/run_all.sh"
