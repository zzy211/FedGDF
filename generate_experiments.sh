#!/bin/bash

# Define parameters
datasets=("Cora" "Citeseer" "Cs" "Physics")
alg_methods=("FedKD")
base_dir="script/plot"
gpu_id=3

# Create base directory
mkdir -p "$base_dir"

# Clear previous running scripts
> "$base_dir/run_all.sh"

# Iterate over all datasets and algorithm methods
for dataset in "${datasets[@]}"; do
    # Create dataset directory
    dataset_dir="$base_dir/$dataset"
    mkdir -p "$dataset_dir"
    
    for alg_method in "${alg_methods[@]}"; do
        # Script file name
        script_name="plot_${alg_method,,}_${dataset,,}.sh"
        script_path="$dataset_dir/$script_name"

        # Write script content
        cat > "$script_path" << SCRIPT
#!/bin/bash
python run_node_task_main.py --dataset $dataset --num_workers 10 --overlapping_rate 0.0 --is_iid iid --alg_method $alg_method --ALPHA 1.0 --lam_real 0 --sample_num 100 --lam2 0 --gpu_id $gpu_id --draw_decision_bound
SCRIPT
        
        # Make the script executable
        chmod +x "$script_path"
        
        # Generate run commands and add them to the main run script
        echo "nohup sh $script_path > $dataset_dir/${script_name%.sh}.log 2>&1 &" >> "$base_dir/run_all.sh"
        
        echo "创建脚本: $script_path"
    done
done

# Make the main run script executable
chmod +x "$base_dir/run_all.sh"

echo "Completed!"
echo "All scripts have been generated in: $base_dir"
echo "To run all experiments, execute: sh $base_dir/run_all.sh"