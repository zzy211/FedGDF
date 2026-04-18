from Node_level_Models.configs.config import args_parser
from Node_level_Models.helpers.metrics_utils import log_test_results

import numpy as np
import wandb
import os
import pandas as pd
# from node_clf import main as node_main
from time import time

args = args_parser()
rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000, size=3)

project_name = [args.proj_name, args.proj_name+ "debug"]
proj_name = project_name[0]

def main(args):
    if args.alg_method == "FedTAD":
        from FedTAD_node_clf import main as node_main  
    elif args.alg_method == "FedGEN":
        from FedGEN_node_clf import main as node_main
    elif args.alg_method == "Fedproto":
        from FedProto_node_clf import main as node_main
    elif args.alg_method == "FGPL":
        from FGPL_node_clf import main as node_main
    elif args.alg_method == "FedKD":    #our method
        from FedKD_node_clf import main as node_main
    elif args.alg_method == "MultiFedKD":
        from MultiFedKD_node_clf import main as node_main
    elif args.alg_method == "Local":
        from Local_node_clf import main as node_main
    elif args.alg_method == "FedKD_low_cost":
        from FedKD_low_cost_v3 import main as node_main
    elif args.alg_method == "FedTGP":
        from FedTGP_node_clf import main as node_main
    else:
        raise ValueError("alg_method is not defined!")

    model_name = args.model
    Alg_name = "Alg-" + args.alg_method
    file_name = Alg_name + 'Dataset-{}_Model-{}_IID-{}_Num_client-{}_Over_rate-{}'.format(
        args.dataset,
        model_name,
        args.is_iid,
        args.num_workers,
        args.overlapping_rate)

    average_overall_performance_list = []
    average_round_reach_target_list = []
    average_max_acc_list = []
    average_f1_score_list = []
    results_table = []
    metric_list = []
    
    for i in range(len(seeds)): #average the result of 5 tests
        args.seed = seeds[i]
        os.environ["WANDB_MODE"] = "offline" #offline or online
        # wandb init
        if args.alg_method == "BNS_GCN":
            log_dir = os.path.join(
            'wandb', 
            args.model,
            args.dataset,
            f"alg_method_{args.alg_method}_ratio_node_{args.ratio_node}",
            f"num_workers_{args.num_workers}_dirichlet_alpha_{args.dirichlet_alpha}_round_{i}"
            )
        else:
            log_dir = os.path.join(
            'wandb', 
            args.model,
            args.dataset,
            f"alg_method_{args.alg_method}",
            f"num_workers_{args.num_workers}_dirichlet_alpha_{args.dirichlet_alpha}_round_{i}"
            )
        os.makedirs(log_dir, exist_ok=True)
        logger = wandb.init(
            project=proj_name,
            group=file_name,
            name=f"round_{i}",
            config=args,
            dir=log_dir
        )

        average_overall_performance, round_reach_target_acc, max_acc, end_f1_score = node_main(args, logger)
        results_table.append([average_overall_performance, round_reach_target_acc, max_acc, end_f1_score])
        logger.log({"average_overall_performance": average_overall_performance, "round_reach_target": round_reach_target_acc, "max_acc": max_acc, "f1_score": end_f1_score})

        average_overall_performance_list.append(average_overall_performance)
        average_round_reach_target_list.append(round_reach_target_acc)
        average_max_acc_list.append(max_acc)
        average_f1_score_list.append(end_f1_score)
        
        # end the logger
        wandb.finish()

    # wandb table logger init
    columns = ["average_overall_performance_acc", "average_round_reach_target", "average_max_acc", "average_f1_score"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger = wandb.init(
        #entity="hkust-gz",
        project=proj_name,
        group=file_name,
        name=f"exp_results",
        config=args,
    )
    table_logger.log({"results": logger_table})
    wandb.finish()

    mean_average_overall_performance, mean_average_round_reach_target, mean_max_accuracy, mean_end_f1_core = np.mean(np.array(average_overall_performance_list)), \
                                                                        np.mean(np.array(average_round_reach_target_list)),\
                                                                        np.mean(np.array(average_max_acc_list)),\
                                                                        np.mean(np.array(average_f1_score_list))

    std_average_overall_performance, std_average_round_reach_target, std_max_accuracy, std_end_f1_score = np.std(np.array(average_overall_performance_list)),\
                                                                      np.std(np.array(average_round_reach_target_list)),\
                                                                      np.std(np.array(average_max_acc_list)),\
                                                                      np.std(np.array(average_f1_score_list))

    header = ['dataset', 'model', 'method','num_workers', "mean_average_overall_performance", "std_average_overall_performance", "mean_average_round_reach_target", "std_average_round_reach_target", "mean_average_max_acc", "std_average_max_acc", "mean_average_f1_score", "std_average_f1_score"]
    paths = "./checkpoints/Node/"

    metric_list.append(args.dataset)
    metric_list.append(model_name)
    metric_list.append(args.alg_method)
    metric_list.append(args.num_workers)
    metric_list.append(mean_average_overall_performance)
    metric_list.append(std_average_overall_performance)
    metric_list.append(mean_average_round_reach_target)
    metric_list.append(std_average_round_reach_target)
    metric_list.append(mean_max_accuracy)
    metric_list.append(std_max_accuracy)
    metric_list.append(mean_end_f1_core)
    metric_list.append(std_end_f1_score)

    data = {
        "Metric": [
            "Dataset", 
            "Model Name", 
            'Method',
            'Num_Works',
            "Mean Avg Overall Performance", 
            "Std Avg Overall Performance", 
            "Mean Avg Round Reach Target", 
            "Std Avg Round Reach Target",
            "Mean Avg Max Accuracy",
            "Std Avg Max Accuracy",
            "Mean Avg f1-Score",
            "Std Avg f1-Score"
        ],
        "Value": metric_list  
    }

    df = pd.DataFrame(data)

    print(df)

if __name__ == '__main__':
    args = args_parser()
    main(args)