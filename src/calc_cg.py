import os
import json
import glob
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from lvm_prediction import predict

eta = 0.001
num_steps = 500000

# Load the existing processed project labels
cg_file = 'Data/MB_100_subsampled_results/cg.json'
processed_projects = set()

if os.path.exists(cg_file):
    with open(cg_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            processed_projects.add(item['label'])

# Path to the folders containing the data
folders = glob.glob('data_subsampled/*')

# Function to process each project
def process_project(folder, i, processed_projects):
    project = folder.split('/')[-1]  # Extract the project name
    project_id = f'{project}_{i}'
    
    if project_id in processed_projects:
        return None  # Skip processing if already done
    
    dataset = pd.read_csv(f'{folder}/{project}_{i}.csv', index_col=0).values

    if dataset.shape[0] < 100:
        return f"{project_id} has too few datapoints"
    
    cg_list = []
    cg_null_list = []
    for _ in range(10):
        data_idxs = np.random.choice(dataset.shape[0], 100, replace=False)
        data = dataset[data_idxs]

        mean_abus = data.mean(axis=0)

        data = data[:, mean_abus > 1e-3]
        data = data/data.sum(axis=1)[:, np.newaxis]

        obj = predict(data=data, eta=eta, num_steps=num_steps, plot_loss=False)
    
        try:
            obj.cg_score()
            cg = obj.cg[0]
        except AssertionError:
            cg = np.nan
        
        try:
            obj.cg_score_mean_retain_null()
            cg_null = obj.cg_null[0]
        except AssertionError:
            cg_null = np.nan
    
        cg_list.append(cg)
        cg_null_list.append(cg_null)

    cg_data = {
        'label': project_id,
        'exp_cg': np.mean(cg_list),
        'mean_retain_null': np.mean(cg_null_list)
    }
    
    with open(cg_file, 'a') as f:
        f.write(json.dumps(cg_data) + '\n')
    
    processed_projects.add(project_id)
    
    return f"Processed project {project_id} and saved the best data."

# Iterate through each folder and parallelize the inner loop
for folder in folders:
    results = Parallel(n_jobs=-1)(delayed(process_project)(folder, i, processed_projects) for i in range(10))
    
    # Print results if any
    for result in results:
        if result is not None:
            print(result)