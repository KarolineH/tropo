import pheno4d_util as p4d
import numpy as np
from minleaf.process_leaf import Leaf
import os

'''PARAMETERS'''
output_dir = '/home/karoline/workspace/data/tropo_output/Maize'
data_set = 'Maize' #'Tomato' #'Maize' #'Katrina' #'Zara' #'Zombies'
use_only_annontated = True
recompute = [False, False, False, False]
hyperparams = {
    'ball_pivot': True, 'max mesh edge length': 0.75, 'bp_radii': [0.1,0.15,0.2,0.25,0.3,0.35,0.4],'hole closing max edges':30,
    'target compression percentage': 0.2, 'max_outline_edge_length': 0
}


'''INPUT DATA LOCATIONS'''
if data_set == 'Tomato' or data_set == 'Maize':
    input_dir = '/home/karoline/workspace/data/Pheno4D'
elif data_set == 'Katrina':
    input_dir = '/home/karoline/workspace/data/EinScan/annotated/Katrina/Katrina2/xyz'
elif data_set == 'Zara':
    input_dir = '/home/karoline/workspace/data/EinScan/annotated/Zara/Zara1/xyz'
elif data_set == 'Zombies':
    input_dir = '/home/karoline/workspace/data/zombies_raw'


'''TUNED PARAMETERS'''
#[0.01,0.1,1,1.2] Strawberry default
#[0.1,0.15,0.2,0.25,0.3,0.35,0.4] Zombie BP meshing
#[0.06, 0.1, 0.5, 1] Tomato
# Maize


'''CODE'''
def process_p4d_data(input_dir, output_dir, use_only_annontated, recompute, hyperparams):
    all_files, annotated_files = p4d.get_file_locations(input_dir)
    if use_only_annontated:
        files = annotated_files
    tomato_plants = [scan for sublist in files for scan in sublist if 'Tomato' in scan]
    maize_plants = [scan for sublist in files for scan in sublist if 'Maize' in scan]
    all_plants = (tomato_plants, maize_plants)

    if data_set == 'Tomato':
        plants = all_plants[0]
    elif data_set == 'Maize':
        plants = all_plants[1]

    plants = sorted(plants) # make sure the files are grouped by plant and sorted in order of time
    last_plant = 0
    current_ts = 0 # to calculate recording time steps
    for scan in plants:
        cloud, labels, scan_id = p4d.open_file(scan)
        if data_set == 'Maize':
            labels = labels[:,1] # select the second column of labels
        organs = p4d.split_into_organs(cloud, labels)
        if data_set == 'Tomato':
            leaves = organs[2:]
        elif data_set == 'Maize':
            leaves = organs[1:]

        # Make sure to ID the plant and scan correctly
        plant_nr = int(scan_id.split('_')[0][1:])
        if plant_nr == last_plant:
            current_ts += 1
        else:
            last_plant = plant_nr
            current_ts = 0
        date = scan_id.split('_')[1]

        leaf_labels = np.asarray([int(leaf[1][0]) for leaf in leaves])
        if data_set == 'Tomato':
            leaf_numbers = leaf_labels - 2
        elif data_set == 'Maize':
            leaf_numbers = leaf_labels - 1

        for i,leaf in enumerate(leaves):
            full_id = f"{plant_nr}_{date}_{current_ts}_{leaf_numbers[i]}"
            print(full_id)

            min_leaf = Leaf(leaf[0], full_id, base_dir=output_dir, recompute=recompute, hyperparams=hyperparams)
    return


def process_strawb_data(input_dir, output_dir, recompute, hyperparams):
    files = os.listdir(input_dir)
    for scan in files:
        # This is performed for each individual scan
        data = np.loadtxt(input_dir + '/' + scan)
        class_cats = {"leaf":1, "petiole":2, "berry":3, "flower":4, "crown":5, "background":6}
        points = []
        colours = []
        labels = []
        for entry in data:
            # Get all the leaf points
            x,y,z,r,g,b,semantic_label,instance_label = entry
            if semantic_label == class_cats["leaf"]:
                points.append([x,y,z])
                colours.append([r,g,b])
                labels.append(instance_label)

        points = np.array(points)
        instances = np.unique(labels)

        individuals = []
        for inst in instances:
            ind = np.argwhere(labels == inst).squeeze()
            individuals.append(points[ind])
        
        instance_labels = instances.astype(int)
        for i,leaf in enumerate(individuals):
            full_id = f"{scan.split('.')[0]}_{instance_labels[i]}"
            min_leaf = Leaf(leaf, full_id, base_dir=output_dir, recompute=recompute, hyperparams=hyperparams)
    return

def process_zombie_data(input_dir, output_dir, recompute, hyperparams):
    leaves = os.listdir(input_dir)
    leaves.sort()
    pcs = []
    ids = []
    files = []
    for i,leaf in enumerate(leaves):
        data = np.loadtxt(input_dir + '/' + leaf)
        pc = data[:,:3]
        pcs.append(pc)
        ids.append(''.join(c for c in leaf.split('_')[0] if c.isdigit()) + '_' + ''.join(c for c in leaf.split('_')[1] if c.isdigit()))
        files.append(input_dir + '/' + leaf)
    
    for cloud,nr in zip(pcs,ids):
        min_leaf = Leaf(cloud, nr, base_dir=output_dir, recompute=recompute, hyperparams=hyperparams)
    return

if data_set == 'Tomato' or data_set == 'Maize':
    process_p4d_data(input_dir, output_dir, use_only_annontated, recompute, hyperparams)
if data_set == 'Katrina' or data_set == 'Zara':
    process_strawb_data(input_dir, output_dir, recompute, hyperparams)
if data_set == 'Zombies':
    process_zombie_data(input_dir, output_dir, recompute, hyperparams)