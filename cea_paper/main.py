import os
import numpy as np
import util
import pheno_extraction
import cea_plotting
import leaf_matching

# PREREQUISITES
in_dir = '/home/karo/ws/data/4DBerry/'
out_dir = '/home/karo/ws/data/4DBerry/processed/'
files = util.get_annotated_scans(in_dir)
dates = [str(int(scan.split('_')[1]))[4:] for scan in files]
labels, plant_names = util.scan_file_names_to_labels(files)
leaf_labels, __ = util.leaf_file_names_to_labels(in_dir + 'processed/aligned/')
# util.preprocess_leaves(files, out_dir)


#GET ORGAN COUNTS OVER TIME
# if os.path.exists('cea_paper/organ_counts.csv'):
#     counts = np.loadtxt('cea_paper/organ_counts.csv', delimiter=',')
# else:
#     counts = pheno_extraction.count_organs(files)
# cea_plotting.plot_organ_counts_over_time(counts, labels, dates, plant_names)


# GET BIOMASS OVER TIME
# if os.path.exists('cea_paper/plant_biomass.csv'):
#     biomass = np.loadtxt('cea_paper/plant_biomass.csv', delimiter=',')
# else:
#     biomass = pheno_extraction.biomass(files,voxel_size=2)
# cea_plotting.plot_biomass_over_time(biomass, labels, dates, plant_names)

# GET LEAF MATCHING OVER TIME
centroids = leaf_matching.get_centroids(in_dir + 'processed/transform_log/')
associated_centroids, associated_labels = util.filter_subset(centroids, leaf_labels, plant_nr=0, scan_nr=[0,1,2,3,4])
# scores, nr_of_leaves, pairings_calculated, total_true_pairings, cost_distr = leaf_matching.get_bonn_scores(associated_centroids, associated_labels)
# print(scores) # x, fp, o, dist_x, dist_fp, dist_o
predictions,__,__ = leaf_matching.get_bonn_predictions(centroids,leaf_labels)
# predictions 0 to 4 are valid, 5 is not because it represents the jump in time, 6 is the last prediction for A2
# predictions 7 is valid, 8 is not, 9 to 11 are valid again for B1


# GET LEAF AREA OVER TIME
#pheno_extraction.leaf_area(in_dir)
leaf_labels, __ = util.leaf_file_names_to_labels(in_dir + 'processed/aligned/')
if os.path.exists('cea_paper/leaf_areas.csv'):
    areas = np.loadtxt('cea_paper/leaf_areas.csv', delimiter=',')
    print(areas)
else:
    pheno_extraction.leaf_area(in_dir)
cea_plotting.plot_leaf_area_over_time(areas, leaf_labels, dates, plant_names)
#cea_plotting.plot_area_method_comparison(areas, leaf_labels, dates, plant_names)


# SHOW THE SAME LEAF OVER TIME
leaf_dir = '/home/karo/ws/data/4DBerry/processed/aligned/'
leaf_files = os.listdir(leaf_dir)
leaf_files = sorted(leaf_files)
cea_plotting.plot_single_leave_scans_over_time(leaf_files, leaf_labels)
