import os

# ===================== Configurable Parameters =====================
DATASETS = ["Reddit"]
ALPHA_LIST = [10.0]
NUM_CLIENTS = 20
GPU_ID = 5

# Algorithm configuration: algorithm name + non-iid specific parameters + iid specific parameters
ALGORITHMS = {
    "Local": {
        "non_iid_params": "",
        "iid_params": ""
    },
    "Fedproto": {
        "non_iid_params": "--w_proto 0.5",
        "iid_params": "--w_proto 0.5"
    },
    "FedTAD": {
        "non_iid_params": "--ALPHA 1.0 --sample_num 100 --lam2 1 --topk 5",
        "iid_params": "--ALPHA 1.0 --sample_num 100 --lam2 1 --topk 5"
    },
    "FedTGP": {
        "non_iid_params": "--w_proto 0.5",
        "iid_params": "--w_proto 0.5"
    },
    "FedKD": {
        "non_iid_params": "--ALPHA 1.0 --lam_real 0 --sample_num 100 --lam2 0",
        "iid_params": "--ALPHA 1.0 --lam_real 0 --sample_num 100 --lam2 0"
    },
    "FGPL": {
        "non_iid_params": "",
        "iid_params": ""
    }
}
# ======================================================

def generate_non_iid_scripts():
    """Generate non-iid Dirichlet scripts and run commands"""
    print("=" * 60)
    print("Start generating non-iid Dirichlet run scripts")
    print("=" * 60)
    all_run_commands = []  # Store all nohup commands
    
    for alpha in ALPHA_LIST:
        for dataset in DATASETS:
            for alg, params in ALGORITHMS.items():
                # 1. Define paths and file names
                base_dir = f"script/non_iid/alpha_{alpha}"
                alg_lower = alg.lower()
                script_name = f"{alg_lower}_{alpha}_{dataset.lower()}.sh"
                script_path = os.path.join(base_dir, script_name)
                log_path = script_path.replace(".sh", ".log")
                
                # 2. Create folders
                os.makedirs(base_dir, exist_ok=True)
                
                # 3. Concatenate commands
                command = (
                    f"python run_node_task_main.py --dataset {dataset} --num_workers {NUM_CLIENTS} "
                    f"--overlapping_rate 0.0 --is_iid non-iid-dirichlet --same_size_dataset 1 "
                    f"--dirichlet_alpha {alpha} --alg_method {alg} "
                    f"{params['non_iid_params']} --gpu_id {GPU_ID}"
                )
                
                # 4. Write to the sh script
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write("#!/bin/bash\n")
                    f.write(command + "\n")
                
                # 5. Generate background running commands
                run_command = f"nohup sh {script_path} > {log_path} 2>&1 &"
                all_run_commands.append(run_command)
                
                # 6. Print output
                print(f"\n✅ Script generated: {script_path}")
                print("-" * 50)
    
    return all_run_commands

def generate_batch_run_script(commands):
    """Generate a one-click script to run all tasks"""
    batch_file = "script/non_iid/run_non_iid_all.sh"
    os.makedirs("script/non_iid", exist_ok=True)
    
    with open(batch_file, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# One-click start of all non-iid training tasks\n")
        f.write("echo \"========== Start training all algorithms ==========\"\n")
        for cmd in commands:
            f.write(cmd + "\n")
        f.write("echo \"========== All tasks started in background ==========\"\n")
    
    # Grant execution permissions
    os.chmod(batch_file, 0o755)
    print(f"\n🎉 One-click startup script generated successfully: {batch_file}")
    print(f"🚀 Run command: sh {batch_file}")

def generate_iid_scripts():
    """Optional: Generate IID scripts (according to your needs)"""
    print("\n" + "=" * 60)
    print("Start generating IID run scripts")
    print("=" * 60)
    
    base_dir = "script/iid"
    os.makedirs(base_dir, exist_ok=True)
    
    for dataset in DATASETS:
        for alg, params in ALGORITHMS.items():
            script_name = f"{alg.lower()}_iid_{dataset.lower()}.sh"
            script_path = os.path.join(base_dir, script_name)
            
            command = (
                f"python run_node_task_main.py --dataset {dataset} --num_workers {NUM_CLIENTS} "
                f"--overlapping_rate 0.0 --is_iid iid --alg_method {alg} "
                f"{params['iid_params']} --gpu_id {GPU_ID}"
            )
            
            with open(script_path, "w", encoding="utf-8") as f:
                f.write("#!/bin/bash\n")
                f.write(command + "\n")
            
            run_command = f"nohup sh {script_path} > {script_path.replace('.sh','.log')} 2>&1 &"
            print(f"\n✅ IID script generated: {script_path}")
            print(f"📝 Run command: {run_command}")

if __name__ == "__main__":
    # 1. Generate all non-iid scripts and collect commands
    commands = generate_non_iid_scripts()
    # 2. Generate one-click startup script
    generate_batch_run_script(commands)
    
    print("\n✅ All tasks generated successfully!")