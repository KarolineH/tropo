import numpy as np
import matplotlib.pyplot as plt
import util
import open3d as o3d
import copy

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
        plt.title(f"Organ counts for plant {plant_names[plant]} across time", fontsize=16)
        plt.xlabel("Scan date", fontsize=10)
        plt.ylabel("Number of organs", fontsize=10)
        plt.plot(sublabels[:,1], subset[:,[0,1,2,3,8]], 'o-', marker='o', linewidth=2, markersize=8)
        if plant_names[plant] == 'A2':
            plt.xticks(A2_int_dates, [date[2:]+'.'+date[:2] for date in all_scan_dates_A2], rotation=45, ha='right')
        elif plant_names[plant] == 'B1':
            plt.xticks(B1_int_dates, [date[2:]+'.'+date[:2] for date in all_scan_dates_B1], rotation=45, ha='right')
        else:
            plt.xticks(sublabels[:,1], [date[2:]+'.'+date[:2] for date in subdates.squeeze()],  rotation=45, ha='right')
        plt.legend(["leaf", "petiole", "berry", "flower", "emerging leaf"])
        plt.subplots_adjust(bottom=0.2)
        plt.show()
    return

def plot_biomass_over_time(mass, labels, dates, plant_names):

    all_scan_dates_A2 = ['0512','0519','0525','0531','0608','0616','0620','0623','0627','0629','0707','0715','0721','0729']
    all_scan_dates_B1 = ['0513','0520','0526','0531','0609','0617','0620','0623','0627','0630','0707','0715','0721','0729']
    A2_int_dates = util.get_date_nrs(all_scan_dates_A2)
    B1_int_dates = util.get_date_nrs(all_scan_dates_B1)

    plants = np.unique(labels[:,0])
    plt.figure(figsize=(14, 6))
    #plt.title(f" for over time")
    plt.xlabel("Scan date", fontsize=12)
    plt.ylabel("Estimated plant volume [cm\u00b3]", fontsize=12)

    for plant in plants:
        subset,sublabels = util.filter_subset(mass, labels, plant_nr=plant)
        subdates, sublabels = util.filter_subset(np.asarray(dates).reshape(-1,1), labels, plant_nr=plant)
        subset = subset/1000 # convert from mm^3 to cm^3
        if plant_names[plant] == 'A2':
            plt.plot(sublabels[:,1], subset, 'o-', marker='o', linewidth=2, markersize=8)
        elif plant_names[plant] == 'B1':
            plt.plot(sublabels[:,1] + 1, subset, 'o-', marker='o', linewidth=2, markersize=8)

    B1_int_dates = np.asarray(B1_int_dates) + 1
    xticks = np.unique(np.concatenate((np.asarray(all_scan_dates_A2), np.asarray(all_scan_dates_B1)),0))
    plt.xticks(np.unique(np.concatenate((A2_int_dates,B1_int_dates),0)), [date[2:]+'.'+date[:2] for date in xticks], rotation=90, ha='center')
    line_colors = [line.get_color() for line in plt.gca().lines]
    for i, label in enumerate(plt.gca().get_xticklabels()):
        if xticks[i] in all_scan_dates_A2 and xticks[i] in all_scan_dates_B1:
            label.set_color('black')
        elif xticks[i] in all_scan_dates_A2:
            label.set_color(line_colors[0])
        else:
            label.set_color(line_colors[1])
    plt.legend(["A2", "B1"])
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    return

def plot_single_leave_scans_over_time(files, labels):
    directory = '/home/karo/ws/data/4DBerry/processed/aligned/'

    for plant in np.unique(labels[:,0]):
        for leaf in np.unique(labels[:,-1]):
            subset, sublabels = util.filter_subset(np.asarray(files), labels, plant_nr=plant, leaf_nr=leaf, scan_nr=[0,1,2,3,4]) # For now limited to the first 5 scans

            geometries = []

            for i, scan in enumerate(subset):
                pc = o3d.io.read_point_cloud(directory+scan, format='xyz')
                if np.asarray(pc.points).shape[0] > 0:
                    geometries.append(pc)
            
            if len(geometries) > 0:
                draw_staggered_clouds(geometries, offset=True, labels=sublabels[:,1])
    return

def draw_staggered_clouds(geometries, name='name', offset = False, labels=None):
    '''
    Takes Open3D gemoetry objects and plots them using the o3D visualization.gui.
    Version 2 with improved offset calculation.
    '''
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(name, 1024, 768)
    vis.show_settings = True
    width = 0
    for i,geo in enumerate(geometries):
        geocopy = copy.deepcopy(geo)
        try:
            if offset:
                if i!=0:
                    width += 0.6 * (geometries[i-1].get_max_bound()[0] - geometries[i-1].get_min_bound()[0]) + 0.6 * (geometries[i].get_max_bound()[0] - geometries[i].get_min_bound()[0])
                geocopy.translate((width,0,0))
            vis.add_geometry("{}".format(i),geocopy)
            if labels is not None:
                vis.add_3d_label(geocopy.get_min_bound(),"{}".format(labels[i]))
            # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=geo.get_max_bound()[0] - geo.get_min_bound()[0], origin=geocopy.get_min_bound())
            # vis.add_geometry("box{}".format(i),mesh_frame)
        except:
            pass
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


def plot_leaf_area_over_time(areas, labels, dates, plant_names):
    # for each plant, 4 plots, one using each method
    # the x axis is the date
    # the y axis is the leaf area
    # each leaf is a line

    # Other idea: put the ground truth as solid lines and the predictions as dashed or less opaque lines (e.g. alpha=0.5)
    
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
        using = [0] # not showing MinMesh or Delaunay for now
        for i in using:
            plt.figure()
            plt.title(f"Leaf area for plant {plant} over time")
            plt.xlabel("Scan date")
            plt.ylabel("Leaf area [cm\u00b2]")
            leaf_nrs = []
            for leaf in np.unique(sublabels[:,-1]):
                leaf_set, leaf_labels = util.filter_subset(subset, sublabels, leaf_nr=leaf)
                if len(leaf_set.shape)==1:
                    continue
                leaf_set = leaf_set/100 # convert from mm^2 to cm^2
                plt.plot(leaf_labels[:,1], leaf_set[:,i], 'o-')
                leaf_nrs.append(leaf)
            if plant_names[plant] == 'A2':
                plt.xticks(A2_int_dates, [date[2:]+'.'+date[:2] for date in all_scan_dates_A2], rotation=45, ha='right')
            elif plant_names[plant] == 'B1':
                plt.xticks(B1_int_dates, [date[2:]+'.'+date[:2] for date in all_scan_dates_B1], rotation=45, ha='right')
            plt.legend(leaf_nrs)
            plt.show()
    return

def plot_area_method_comparison(areas, labels, dates, plant_names):
    # for each leaf a plot
    # the x axis is the date
    # the y axis is the leaf area
    # each line is a different method
    
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
        for leaf in np.unique(sublabels[:,-1]):
            if leaf != 6:
                continue
            leafset, leafsublabels = util.filter_subset(subset, sublabels, leaf_nr=leaf)
            if len(leafset.shape)==1:
                continue

            plt.figure()
            plt.title(f"Leaf area for plant {plant_names[plant]}, leaf {leaf} over time")
            plt.xlabel("Scan date")
            plt.ylabel("Leaf area [cm\u00b2]")
            plt.subplots_adjust(bottom=0.2)
            leafset = leafset/100 # convert from mm^2 to cm^2

            method = ['Zabawa', 'Minimal mesh', 'Ball pivoting', 'Delaunay']
            using = [0,2,3] # not showing MinMesh for now
            for i in using:
                plt.plot(leafsublabels[:,1], leafset[:,i], 'o-',  marker='o', linewidth=2, markersize=8)
            plt.legend(np.asarray(method)[using])
            if plant_names[plant] == 'A2':
                if leaf == 6:
                    gt = [36.50,89.45,118.96,146.79,157.45] # in silico ground truth for leaf 6
                    plt.plot(leafsublabels[:,1], gt, 'o-', marker='o', linewidth=2, markersize=8)
                    plt.legend(np.asarray(method)[using].tolist() + ['Ground truth'])
                plt.xticks(A2_int_dates, [date[2:]+'.'+date[:2] for date in all_scan_dates_A2], rotation=45, ha='right')
            elif plant_names[plant] == 'B1':
                plt.xticks(B1_int_dates, [date[2:]+'.'+date[:2] for date in all_scan_dates_B1], rotation=45, ha='right')
            plt.show() 
    return

