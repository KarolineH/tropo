import numpy as np
import matplotlib.pyplot as plt
import util

def plot_organ_counts_over_time(counts, labels, dates, plant_names):

    all_scan_dates_A2 = ['0512','0519','0525','0531','0608','0616','0620','0623','0627','0629','0707','0715','0721','0729']
    all_scan_dates_B1 = ['0513','0520','0526','0531','0609','0617','0620','0623','0627','0630','0707','0715','0721','0729']
    A2_int_dates = util.get_date_nrs(all_scan_dates_A2)
    B1_int_dates = util.get_date_nrs(all_scan_dates_B1)

    plants = np.unique(labels[:,0])
    for plant in plants:
        subset,sublabels = util.filter_subset(counts, labels, plant_nr=plant)
        subdates, sublabels = util.filter_subset(np.asarray(dates).reshape(-1,1), labels, plant_nr=plant)

        plt.figure(figsize=(12, 6))
        plt.title(f"Organ counts for plant {plant_names[plant]} over time")
        plt.xlabel("Scan date")
        plt.ylabel("Number of organs")
        plt.plot(sublabels[:,1], subset[:,[0,1,2,3,8]], 'o-')
        if plant_names[plant] == 'A2':
            plt.xticks(A2_int_dates, [date[2:]+'.'+date[:2] for date in all_scan_dates_A2], rotation=45, ha='right')
        elif plant_names[plant] == 'B1':
            plt.xticks(B1_int_dates, [date[2:]+'.'+date[:2] for date in all_scan_dates_B1], rotation=45, ha='right')
        else:
            plt.xticks(sublabels[:,1], [date[2:]+'.'+date[:2] for date in subdates.squeeze()],  rotation=45, ha='right')
        plt.legend(["leaf", "petiole", "berry", "flower", "emerging leaf"])
        plt.show()
    return

def plot_leaf_area_over_time(areas, labels, dates, plant_names):
        # for each plant, 4 plots, one using each method
        # the x axis is the date
        # the y axis is the leaf area
        # each leaf is a line
    
    all_scan_dates_A2 = ['0512','0519','0525','0531','0608']
    all_scan_dates_B1 = ['0513','0520']
    A2_int_dates = util.get_date_nrs(all_scan_dates_A2)
    B1_int_dates = util.get_date_nrs(all_scan_dates_B1)

    for plant in np.unique(labels[:,0]):
        # Not all annotated scans have had their leaf identities associated over time (manually) yet.
        if plant_names[plant] == 'A2':
                associated_dates = np.arange(5)
        elif plant_names[plant] == 'B1':
                associated_dates = np.arange(2)

        subset, sublabels = util.filter_subset(areas, labels=labels, plant_nr=plant, scan_nr=associated_dates)
        method = ['Zabawa', 'Minimal mesh', 'Ball pivoting', 'Delaunay']
        for i in range(len(method)):
            plt.figure()
            plt.title(f"Leaf area for plant {plant} over time")
            plt.xlabel("Scan date")
            plt.ylabel("Leaf area")
            leaf_nrs = []
            for leaf in np.unique(sublabels[:,-1]):
                leaf_set, leaf_labels = util.filter_subset(subset, sublabels, leaf_nr=leaf)
                if len(leaf_set.shape)==1:
                    continue
                plt.plot(leaf_labels[:,1], leaf_set[:,i], 'o-')
                leaf_nrs.append(leaf)
            if plant_names[plant] == 'A2':
                plt.xticks(A2_int_dates, [date[2:]+'.'+date[:2] for date in all_scan_dates_A2], rotation=45, ha='right')
            elif plant_names[plant] == 'B1':
                plt.xticks(B1_int_dates, [date[2:]+'.'+date[:2] for date in all_scan_dates_B1], rotation=45, ha='right')
            plt.legend(leaf_nrs)
            plt.show()
    return
