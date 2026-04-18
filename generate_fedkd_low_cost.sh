#!/bin/bash

datasets=("Cora" "Citeseer" "Pubmed" "Cs" "Physics")
clients_num=10
base_dir="script/${clients_num}clients"
alg_method="FedKD_low_cost"
gpu_id=0

# Clear previous running scripts
> "$base_dir/run_all.sh"

# Iterate over all datasets
for dataset in "${datasets[@]}"; do
    dataset_dir="$base_dir/$dataset" # Set the correct dataset directory

    if [ ! -d "$dataset_dir" ]; then
        mkdir -p "$dataset_dir"
    else
        echo "Directory $dataset_dir already exists, skipping"
    fi
    
    # Generate script name based on dataset and algorithm method
    script_name="${alg_method,,}_${dataset,,}_${clients_num,,}_v3.sh"
    script_path="$dataset_dir/$script_name"

    cat > "$script_path" << SCRIPT
#!/bin/bash
python run_node_task_main.py --dataset $dataset --num_workers $clients_num --overlapping_rate 0.0 --is_iid iid --alg_method FedKD_low_cost  --ALPHA 1.0 --lam_real 0 --sample_num 100 --lam2 0 --gpu_id 0
SCRIPT

    # Make the script executable
    chmod +x "$script_path"

    # Add the command to the main run script for easy one-click execution
    echo "nohup sh $script_path > $dataset_dir/${script_name%.sh}.log 2>&1 &" >> "$base_dir/run_all.sh"
    echo "Created script: $script_path"
done

# Make the main run script executable
chmod +x "$base_dir/run_all.sh"

echo "Completed!"
echo "All scripts have been generated in: $base_dir"
echo "To run all experiments, execute: sh $base_dir/run_all.sh"
