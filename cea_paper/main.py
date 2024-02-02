import os
import numpy as np
import util
import pheno_extraction
import cea_plotting

# PREREQUISITES
in_dir = '/home/karoline/workspace/data/4DBerry/'
out_dir = '/home/karoline/workspace/data/4DBerry/processed/'
files = util.get_annotated_scans(in_dir)
dates = [str(int(scan.split('_')[1]))[4:] for scan in files]
labels, plant_names = util.scan_file_names_to_labels(files)

# GET ORGAN COUNTS OVER TIME
if os.path.exists('cea_paper/organ_counts.csv'):
    counts = np.loadtxt('cea_paper/organ_counts.csv', delimiter=',')
else:
    counts = pheno_extraction.count_organs(files)
#cea_plotting.plot_organ_counts_over_time(counts, labels, dates, plant_names)


# GET LEAF AREA OVER TIME
# pheno_extraction.leaf_area(in_dir)
leaf_labels, __ = util.leaf_file_names_to_labels(in_dir + 'processed/aligned/')
if os.path.exists('cea_paper/leaf_areas.csv'):
    areas = np.loadtxt('cea_paper/leaf_areas.csv', delimiter=',')
    print(areas)
else:
    pheno_extraction.leaf_area(in_dir)
cea_plotting.plot_leaf_area_over_time(areas, leaf_labels, dates, plant_names)