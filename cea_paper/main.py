import os
import numpy as np
import util
import pheno_extraction
import cea_plotting
import leaf_matching

# PREREQUISITES
in_dir = '/home/karoline/workspace/data/4DBerry/'
out_dir = '/home/karoline/workspace/data/4DBerry/processed/'
files = util.get_annotated_scans(in_dir)
dates = [str(int(scan.split('_')[1]))[4:] for scan in files]
labels, plant_names = util.scan_file_names_to_labels(files)
# util.preprocess_leaves(files, out_dir)


# GET ORGAN COUNTS OVER TIME
# if os.path.exists('cea_paper/organ_counts.csv'):
#     counts = np.loadtxt('cea_paper/organ_counts.csv', delimiter=',')
# else:
#     counts = pheno_extraction.count_organs(files)
#cea_plotting.plot_organ_counts_over_time(counts, labels, dates, plant_names)


# GET LEAF AREA OVER TIME
# pheno_extraction.leaf_area(in_dir)
leaf_labels, __ = util.leaf_file_names_to_labels(in_dir + 'processed/aligned/')
# if os.path.exists('cea_paper/leaf_areas.csv'):
#     areas = np.loadtxt('cea_paper/leaf_areas.csv', delimiter=',')
#     print(areas)
# else:
#     pheno_extraction.leaf_area(in_dir)
# cea_plotting.plot_leaf_area_over_time(areas, leaf_labels, dates, plant_names)
#cea_plotting.plot_area_method_comparison(areas, leaf_labels, dates, plant_names)

# GET LEAF MATCHING OVER TIME
centroids = leaf_matching.get_centroids(in_dir + 'processed/transform_log/')
associated_centroids, associated_labels = util.filter_subset(centroids, leaf_labels, plant_nr=0, scan_nr=[0,1,2,3,4])
scores, nr_of_leaves, pairings_calculated, total_true_pairings, cost_distr = leaf_matching.get_bonn_scores(associated_centroids, associated_labels)
print(scores) # x, fp, o, dist_x, dist_fp, dist_o