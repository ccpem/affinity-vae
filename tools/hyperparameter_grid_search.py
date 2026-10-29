#Created by Anne Sekscinska, October 2026
import os
import itertools
import subprocess
import shutil
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
EXAMPLE USE 

config = "configs/avae-test-config.yml"
beta_values = [0.01,0.1,0.5,1,1.5,2, 2.5]
gamma_values = [0.01,0.1,0.5,1,1.5,2, 2.5]
grid_search_output = "./gs_res"
eval_output = "./gs_eval"

hyperparameter_grid_search(config, beta_values, gamma_values, grid_search_output)
evaluate_grid_search(grid_search_output, eval_output)

"""


def hyperparameter_grid_search(config: str, beta_values: List[float], gamma_values: List[float], output_path: str = "./"):
    """A wrapper script to perform a hyperparameter for beta and gamma values for affinityVAE. Creates an output folder with subfolders with run names,
    populated with results of affinityVAE for each run.
    
    Please note the script assumes the current working directory is the main affinityVAE directory.
    If this is not the case change the directory of run.py in  line 46 of this script.
    
    Parameters
    ----------
    config: str
        Path to the basic affinityVAE for which the parameters are to be optimised
    beta_values: List[float]
        List of beta values, recommended between 0.01 to 2.5 with increment of 0.5
    gamma_values: List[float]
        List of gamma values, recommended between 0.01 to 2.5 with increment of 0.5
    output path: str
        Path to the output folder for all results, default is the current directory
    """
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/gs_logs", exist_ok=True)
    
    for beta, gamma in itertools.product(beta_values, gamma_values):
        
        #Set up the run folder
        run_name = f"beta{beta}_gamma{gamma}"
        run_dir  = os.path.join(output_path, run_name)
        
        #To control for repeat runs now the script marks something as complete if a folder with a corresponding run name exists
        if not os.path.exists(run_dir):
            print(f"Running affinity for beta: {beta} and gamma: {gamma}")
            
            #It is assumed the working directory is the affinityVAE directory, please change the cmd line below if it is needed to adjust where run.py is
            cmd = ["python", "run.py", 
            "--config_file", config, 
            "--beta", str(beta),
            "--gamma", str(gamma)]

            #Keep logs of each run 
            log_file = f"gs_logs/beta{beta}_gamma{gamma}.log"
            with open(log_file, "w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=log)
                
            #Move the plots, models, and latents to the run directory and clean up
            if os.path.exists("plots"):
                shutil.move("plots", run_dir)
            if os.path.exists("states"):
                shutil.move("states", run_dir)
            if os.path.exists("latents"):
                shutil.move("latents", run_dir)
            shutil.copy(log_file, os.path.join(run_dir, "run.log"))
            print(f"Run done for beta: {beta}, gamma: {gamma} ")
            
        else:
            print(f"Run already done {run_name}")
            continue



def evaluate_grid_search(results_path:str, output_path:str):
    """Evaluates the grid search results by going through subfolders with each run. Saves a .csv file with statistics and a heatmap of last average F1 per run.
    
    Parameters
    ----------
    results_path: str 
            Path to grid search results, the same as in the hyperparameter_grid_search function
    output_path: str
            Path to save the results
    """
    
    f1_stats = {}
    other_stats =  {}
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    #Loop through each run
    for run_dir in os.listdir(results_path):
    
        #save the hyperparameter values
        beta, gamma = (run_dir.split("_"))
        beta_val = float(beta.replace("beta", ""))
        gamma_val = float(gamma.replace("gamma", ""))
        
        #Find the F1 and save statistics
        f1_temp = pd.read_csv(f"{results_path}/{run_dir}/f1_valid.csv")
        avg_f1 = f1_temp["f1_avg"]
        max_key = max(avg_f1.keys())
        f1_stats[run_dir] = {"beta":beta_val, "gamma":gamma_val, "f1_avg_last":avg_f1[max_key], "f1_best": avg_f1.max(),
        "best_epoch": f1_temp.loc[f1_temp["f1_avg"].idxmax(), "epoch"]}
        
        
    #Save the .csv with statistics
    f1_df = pd.DataFrame.from_dict(f1_stats)
    csv_path = os.path.join(output_path, "grid_search_results.csv")
    f1_df.to_csv(csv_path, index=False)

    #Get the data for the heatmap
    f1_t = f1_df.T.reset_index(drop=True)
    f1_t = f1_t.astype(float)
    heatmap_df = f1_t.pivot(index="beta", columns="gamma", values="f1_avg_last")

    #Make and save heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_df, annot = True)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Beta")

    plt.savefig(f"{output_path}/grid_search_heatmap.png")
    
