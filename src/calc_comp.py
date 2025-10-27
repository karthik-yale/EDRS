import numpy as np
import scipy as sp
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from joblib import Parallel, delayed
import pickle
import glob
import time
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import os
import json

# This function computes the competition for one realization of the preference matrix

# This function computes the competition for one realization of the preference matrix

def compute_iteration(preference_idxs_list, ref_crm):
    '''For each realization of the preference matrix of cow bacterias, we only consider the resources that are used at least by one bacterium'''
    crm_cow = []
    for preference_idxs in preference_idxs_list:
        if np.array(preference_idxs).shape[0] > 1:
            crm_cow.append(ref_crm[preference_idxs[np.random.randint(len(preference_idxs))]].reshape(-1))
        else:
            crm_cow.append(ref_crm[preference_idxs].reshape(-1))
    crm_cow = np.array(crm_cow)
    non_zero_cols = np.sum(crm_cow, axis=0) != 0

    crm_cow = crm_cow[:, non_zero_cols]
    sparsity = np.count_nonzero(crm_cow)/(crm_cow.shape[0]*crm_cow.shape[1])
    expectation = crm_cow.shape[1]*sparsity#[crm_cow.shape[0]*sparsity**k for k in range(2,7)]
    comp = resource_by_species_prob(crm_cow.T)/expectation

    rand_sparsity = np.inf
    dist = np.inf
    counter = 0
    while counter < 500:
        rand_idx = np.random.permutation(ref_crm.shape[0])
        crm_rand_new = ref_crm[rand_idx[:crm_cow.shape[0]]]

        crm_rand_new = crm_rand_new[:, non_zero_cols]
        rand_sparsity_new = np.count_nonzero(crm_rand_new)/(crm_rand_new.shape[0]*crm_rand_new.shape[1])

        new_dist = np.abs(rand_sparsity_new - sparsity)/sparsity

        if new_dist < dist:
            dist = new_dist
            crm_rand = np.copy(crm_rand_new)
            rand_sparsity = rand_sparsity_new

        counter += 1

        if dist < 0.1:
            break
        # print(sparsity, rand_sparsity)
    expectation = crm_rand.shape[1]*rand_sparsity#[crm_rand.shape[0]*sparsity**k for k in range(2,7)]
    comp_rand = resource_by_species_prob(crm_rand.T)/expectation

    return comp, comp_rand

def areDistinct(arr) :

    n = len(arr)

    # Put all array elements in a map
    s = set()
    for i in range(0, n):
        s.add(arr[i])

    # If all elements are distinct,
    # size of set should be same array.
    return (len(s) == len(arr))

def find_common_elements(lists):
    # Check if there are at least two lists to find common elements
    if len(lists) < 2:
        return []

    # Initialize the common elements with the first list
    common_elements = np.copy(lists[0])

    # Iterate through the remaining lists and find common elements
    for l_1 in lists[1:]:
        common_elements = np.intersect1d(common_elements, l_1)

    return common_elements.tolist()

def resource_by_species_prob(mat):
    l = []
    S = mat.shape[1]
    for i in range(2,7):
        pos_list = np.random.randint(0, S, size=(100000, i))
        row = []
        pos_list = np.array([pos for pos in pos_list if areDistinct(pos)])
        for pos in pos_list:
            species = np.array([mat[:, idx] for idx in pos])
            num_common = len(find_common_elements([np.nonzero(s) for s in species])) #Number of resources shared by i species
            row.append(num_common)
        l.append(np.sum(row)/len(row))
    return np.array(l)

def plot_res_by_species(mat):
    #mat size = (num_resources, num_species)
    list_1 = resource_by_species_prob(mat)
    sparsity = np.count_nonzero(mat)/(mat.shape[0]*mat.shape[1])
    expectation = [mat.shape[0]*sparsity**k for k in range(2,7)]

    plt.plot(range(2,7), list_1, label='Given')
    plt.plot(range(2,7), expectation, label='expectation')
    plt.yscale('log')
    plt.xticks(range(2,7))
    plt.xlabel('Number of species')
    plt.ylabel('Expected number of resources shared')
    plt.title('Average number of reources shared by k species')
    plt.legend()
    plt.show()

def percentiles(data):
    p1 = np.percentile(data, 25, axis=0)
    p2 = np.percentile(data, 75, axis=0)
    avg = np.average(data, axis=0)

    return p1.reshape(-1), p2.reshape(-1), avg.reshape(-1)


ref_crm = loadmat('/home/ks2823/Competition_Calc/human_mat_files/prefmat.mat')['prefmat'].A
ref_crm[ref_crm < 1e-10] = 0
ref_crm[ref_crm > 0] = 1

# idxs_file = os.path.join('/home/ks2823/palmer_scratch', 'pref_microbiomap_subsampled_class_idxs.json')
# matching_idxs_lists = []
# project_ids_list = []
# if os.path.exists(idxs_file):
#     with open(idxs_file, 'r') as file:
#         for line in file:
#             item = json.loads(line)
#             matching_idxs_lists.append(item['idxs'])
#             project_ids_list.append(item['label'])

# print('Total no. of projects:', len(project_ids_list))



output_file_path = 'Data/MB_100_subsampled_results/comp_w_cutoff_100.json'

# Initialize set for processed projects
processed_projects = set()

# Check if output file exists and load processed project IDs
if os.path.exists(output_file_path):
    with open(output_file_path, 'r') as file:
        for line in file:
            item = json.loads(line)
            processed_projects.add(item['label'])

input_file_path = os.path.join('/home/ks2823/palmer_scratch', 'pref_microbiomap_subsampled_100_class_idxs.json')

chunk_size = 10000  # Adjust based on memory limits

with pd.read_json(input_file_path, lines=True, chunksize=chunk_size) as reader:
    for chunk in reader:
        print(chunk.head()) 

        for _, row in chunk.iterrows():

            project_id = f'{row['label']}_{row['iteration']}'
            # Skip if project already processed
            if project_id in processed_projects:
                continue

            matching_idxs = row['idxs']

            comp_list_cow, comp_list = zip(*Parallel(n_jobs=-1)(delayed(compute_iteration)(matching_idxs, ref_crm) for _ in range(10)))
            mean_comp_cow = np.mean(comp_list_cow, axis=0)
            mean_comp = np.mean(comp_list, axis=0)
            comp_ratio = mean_comp_cow / mean_comp
            
            # Prepare output dictionary
            output = {
                'label': project_id,
                'comp': comp_ratio.tolist()
            }
            print(output)
            # Write output dictionary to JSON file
            with open(output_file_path, 'a') as file:
                file.write(json.dumps(output) + '\n')