import os
import numpy as np
import matplotlib.pyplot as plt
from minleaf.process_leaf import Leaf
import copy


def get_scans(in_dir):
    files = os.listdir(in_dir)
    files.sort()
    files = [in_dir + scan for scan in files]
    return files

def get_annotated_scans(files):
    annotated_files = [scan for scan in files if '_a' in scan]
    return annotated_files


def count_organs(files):
    all_organ_counts = []
    for scan in files:
        data = np.loadtxt(scan, comments="//")
        class_cats = {"leaf":1, "petiole":2, "berry":3, "flower":4, "crown":5, "background":6, "other":7, "table":8, "emerging leaf":9}
        semantic_labels = data[:,-2]
        instance_labels = data[:,-1]

        num_organs = np.zeros(len(class_cats), dtype=int)
        for cat in class_cats.values():
            idx = np.argwhere(semantic_labels == cat).squeeze()
            instance_labels = data[idx,-1]
            instances, counts = np.unique(instance_labels, return_counts=True)
            num_organs[cat-1] = len(instances)

        all_organ_counts.append(num_organs)
    all_organ_counts = np.asarray(all_organ_counts) # array of shape (num_scans, num_classes)
    np.savetxt('organ_counts.csv', all_organ_counts, delimiter=',', fmt='%d')
    return all_organ_counts

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

def plot_organ_counts_ot(dates_A, counts_A, dates_B, counts_B, dates):
    # only use columns 0:3 and 8
    counts_A = counts_A[:,[0,1,2,3,8]]
    counts_B = counts_B[:,[0,1,2,3,8]]

    plt.figure()
    plt.title("Organ counts for plant A2 over time")
    plt.xlabel("Scan date")
    plt.ylabel("Number of organs")
    plt.plot(dates_A, counts_A, 'o-')
    plt.xticks(dates_A, [date[:2]+'.'+date[2:] for date in dates[:7]])
    plt.legend(["leaf", "petiole", "berry", "flower", "emerging leaf"])
    plt.show()

    plt.figure()
    plt.title("Organ counts for plant B1 over time")
    plt.xlabel("Scan date")
    plt.ylabel("Number of organs")
    plt.plot(dates_B, counts_B, 'o-')
    plt.xticks(dates_B, [date[:2]+'.'+date[2:] for date in dates[7:]])
    plt.legend(["leaf", "petiole", "berry", "flower", "emerging leaf"])
    plt.show()


def filter_subset(data, ids=None, labels=None, plant_nr=None, day=None, leaf_nr=None):
    if labels is None:
        if ids is None:
            print("No labels or ids given")
        labels = np.asarray([(string.split('_')[0],string.split('_')[1][-4:],string.split('_')[-1]) for string in ids])
    subset = copy.deepcopy(data)
    sublabels = copy.deepcopy(labels)

    if plant_nr is not None:
        subset = subset[np.argwhere(sublabels[:,0] == plant_nr).squeeze()]
        sublabels = sublabels[np.argwhere(sublabels[:,0] == plant_nr).squeeze()]
    if day is not None:
        subset = subset[np.argwhere(sublabels[:,1] == day).squeeze()]
        sublabels = sublabels[np.argwhere(sublabels[:,1] == day).squeeze()]
    if leaf_nr is not None:
        subset = subset[np.argwhere(sublabels[:,2] == leaf_nr).squeeze()]
        sublabels = sublabels[np.argwhere(sublabels[:,2] == leaf_nr).squeeze()]
    return subset, sublabels

def ids_to_labels(ids):
    labels = np.asarray([(string.split('_')[0],string.split('_')[1][-4:],string.split('_')[-1]) for string in ids])
    return labels
    
def plot_leaf_areas_ot(dates_A, dates_B, labels, areas, leaf_counts):
    # There are 13 plants in this dataset, I want to make one plot per plant
    # Each plot should have a line for each leaf of the plant
    # The x-axis should be the date
    # The y-axis should be the leaf area

    unique_plant_names = np.unique(labels[:,0])
    for plant in unique_plant_names:
        start_date = ['0512' if 'A' in plant else '0513'][0]
        subset, sublabels = filter_subset(areas, labels=labels, plant_nr=plant)
        unique_leaves = np.unique(sublabels[:,2])
        plt.figure()
        for leaf in unique_leaves:
            y, leaf_labels = filter_subset(subset, labels=sublabels, leaf_nr=leaf)
            if len(leaf_labels.shape) == 1:
                leaf_labels = leaf_labels.reshape(1,-1)
            x = get_date_nrs(leaf_labels[:,1], start_date=start_date)
            # add a line to the plot
            plt.plot(x, y, 'o-')

        plt.title(f"Leaf area for plant {plant}")
        plt.xlabel("Date")
        plt.ylabel("Leaf area")
        plt.show()

        unique_dates = np.unique(labels[:,1])

        plt.figure()
        plt.title(f"Leaf area for leaf {leaf+1} over time")
        plt.xlabel("Scan date")
        plt.ylabel("Leaf area")
        plt.plot(dates_A, areas[:7,leaf], 'o-')
        plt.plot(dates_B, areas[7:,leaf], 'o-')
        plt.xticks(dates_A, [date[:2]+'.'+date[2:] for date in dates[:7]])
        plt.xticks(dates_B, [date[:2]+'.'+date[2:] for date in dates[7:]])
        plt.legend(["plant A", "plant B"])
        plt.show()

    # entry = 0
    # for plantnr,(plant_id,count) in enumerate(zip(ids, leaf_counts)):
    #     idx = np.where([plant_id in entry.replace('_','') for entry in areas[:,0]])
    #     leaf_areas = areas[idx]

    #     plt.figure()
    #     leaf_areas = areas[entry:entry+count]
    #     plt.title(f"Leaf areas for plant {ids[plantnr]} over time")
    #     plt.xlabel("Scan date")
    #     plt.ylabel("Leaf area")
    #     for i in range(plant):
    #         plt.plot(dates, areas[:,i], 'o-')
    #     plt.xticks(dates, [date[:2]+'.'+date[2:] for date in dates])
    #     plt.legend([f"leaf {i+1}" for i in range(plant)])
    #     plt.show()
    

    # plt.figure()
    # plt.title("Leaf area over time")
    # plt.xlabel("Scan date")
    # plt.ylabel("Leaf area")
    # plt.plot(dates, areas, 'o-')
    # plt.xticks(dates, [date[:2]+'.'+date[2:] for date in dates])
    # plt.legend(["leaf", "petiole", "berry", "flower", "emerging leaf"])
    # plt.show()

    return


def preprocess_straw_data(files, out_dir):
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
            min_leaf = Leaf(leaf, full_id, base_dir=out_dir, recompute=recompute) # can also pass hyperparams here
            leaf_objects.append(min_leaf)
    return leaf_objects

in_dir = '/home/karo/ws/data/4DBerry/'
out_dir = '/home/karo/ws/data/4DBerry/processed/'
files = get_scans(in_dir)
files = get_annotated_scans(files)
dates = [str(int(scan.split('_')[1]))[4:] for scan in files]

# GET THE ORGAN COUNTS OVER TIME
# counts = count_organs(files)  
# counts = np.loadtxt('organ_counts.csv', delimiter=',')
# counts_A = counts[:7,:]
# counts_B = counts[7:,:]dates_A = get_date_nrs(dates[:7])
dates_A = get_date_nrs(dates[:7])
dates_B = get_date_nrs(dates[7:])
# plot_organ_counts_ot(dates_A, counts_A, dates_B, counts_B, dates)

# GET LEAF AREA OVER TIME
if os.path.exists('leaf_areas_mesh.csv'):
    array_from_file = np.loadtxt('leaf_areas_mesh.csv', delimiter=',', dtype='str')
    ids = array_from_file[:,0]
    areas = array_from_file[:,1].astype(float)
else:
    leaves = preprocess_straw_data(files, out_dir=out_dir)
    areas = np.asarray([leaf.area for leaf in leaves])
    ids = np.asarray([leaf.ID for leaf in leaves])
    array = np.stack([ids, areas], axis=1)
    np.savetxt('leaf_areas_mesh.csv', array, delimiter=',')

__, leaf_counts = np.unique([''.join(filename.split('_')[:2]) for filename in os.listdir(out_dir+'aligned')], return_counts=True)
labels = ids_to_labels(ids)
plot_leaf_areas_ot(dates_A, dates_B, labels, areas, leaf_counts)

print(leaves[0].area)