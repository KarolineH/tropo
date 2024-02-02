import os
import numpy as np
import sys
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from minleaf.process_leaf import Leaf

def get_scans(in_dir):
    files = os.listdir(in_dir)
    files.sort()
    files = [in_dir + scan for scan in files]
    return files

def get_annotated_scans(in_dir):
    files = get_scans(in_dir)
    annotated_files = [scan for scan in files if '_a' in scan]
    return annotated_files

def scan_file_names_to_labels(files):
    split_names = np.asarray([scan.split('/')[-1].split('.')[0].split('_') for scan in files])
    labels = np.zeros(split_names.shape, dtype=int)
    plants = list(np.unique(split_names[:,0]))

    for i,plant in enumerate(plants):
        idx = np.where(split_names[:,0] == plant)
        labels[idx,0] = i

        dates = [entry[-4:] for entry in split_names[idx,1].squeeze()]
        date_ints = get_date_nrs(dates)
        labels[idx,1] = date_ints
        labels[idx,2] = idx
    return labels, plants

def leaf_file_names_to_labels(directory):
    files = os.listdir(directory)
    files = sorted(files)
    split_names = np.asarray([scan.split('/')[-1].split('.')[0].split('_') for scan in files])
    labels = np.zeros(split_names.shape, dtype=int)
    plants = list(np.unique(split_names[:,0]))

    for i,plant in enumerate(plants):
        idx = np.where(split_names[:,0] == plant)[0]
        labels[idx,0] = i

        dates = [entry[-4:] for entry in split_names[idx,1].squeeze()]
        date_ints = get_date_nrs(dates)
        labels[idx,1] = date_ints
        for j in range(1, labels.shape[0]):
            if labels[j, 1] == 0:
                labels[j, 2] = 0
            elif labels[j, 1] > labels[j - 1, 1]:
                labels[j, 2] = labels[j - 1, 2] + 1
            else:
                labels[j, 2] = labels[j - 1, 2]

    labels[:,-1] = split_names[:,-1]
    return labels, plants

def get_date_nrs(dates, start_date=None):
    '''
    Turn dates into integer day numbers starting from 0 (first scan)
    or optionally define a different start date.
    Works specifically for dates starting in May.
    '''
    if start_date is None:
        start_date = dates[0]
    day_nrs = []
    for date in dates:
        month, day = int(date[:2]), int(date[2:])
        if month == 5:
            day_nrs.append(day-int(start_date[-2:]))
        elif month == 6:
            day_nrs.append(day+(31-int(start_date[-2:])))
        elif month == 7:
            day_nrs.append(day+(31+30-int(start_date[-2:])))
        elif month == 8:
            day_nrs.append(day+(31+30+31-int(start_date[-2:])))
        else:
            print("Month not in range")
    return day_nrs

def filter_subset(data, labels, plant_nr=None, day_nr=None, scan_nr=None, leaf_nr=None):
    
    # can be used with target integers or np arrays of multiple integers

    subset = copy.deepcopy(data)
    sublabels = copy.deepcopy(labels)

    if plant_nr is not None:
        subset = subset[np.argwhere(np.isin(sublabels[:,0], plant_nr)).squeeze()]
        sublabels = sublabels[np.argwhere(np.isin(sublabels[:,0], plant_nr)).squeeze()]
    if day_nr is not None:
        subset = subset[np.argwhere(np.isin(sublabels[:,1], day_nr)).squeeze()]
        sublabels = sublabels[np.argwhere(np.isin(sublabels[:,1], day_nr)).squeeze()]
    if scan_nr is not None:
        subset = subset[np.argwhere(np.isin(sublabels[:,2], scan_nr)).squeeze()]
        sublabels = sublabels[np.argwhere(np.isin(sublabels[:,2], scan_nr)).squeeze()]
    if labels.shape[1] > 3 and leaf_nr is not None:
        subset = subset[np.argwhere(np.isin(sublabels[:,3], leaf_nr)).squeeze()]
        sublabels = sublabels[np.argwhere(np.isin(sublabels[:,3], leaf_nr)).squeeze()]
    return subset, sublabels

def preprocess_leaves(files, out_dir):
    '''
    Given a list of annotated scans, this function will create a Leaf object for each individual leaf.
    Returns the leaf object but also writes all steps to files.
    '''
    hyperparams = {'ball_pivot': True, 
                'max mesh edge length': 0.75, 
                'bp_radius': 0,
                'hole closing max edges':30,
                'target compression percentage': 0.3, 
                'max_hull_edge_length': 0.1,
                'pca_input_n': 500}
    
    recompute = [False, False, False, False, False]

    leaf_objects = []
    for scan in files:
        # This is performed for each individual scan
        data = np.loadtxt(scan, comments="//")
        class_cats = {"leaf":1, "petiole":2, "berry":3, "flower":4, "crown":5, "background":6, "other":7, "table":8, "emerging leaf":9}
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
            full_id = f"{scan.split('/')[-1].split('.')[0]}_{instance_labels[i]}"
            min_leaf = Leaf(leaf, full_id, base_dir=out_dir, recompute=recompute, hyperparams=hyperparams)
            leaf_objects.append(min_leaf)
    return leaf_objects