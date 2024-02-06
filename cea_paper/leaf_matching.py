import os
import numpy as np
import util
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def get_centroids(directory):
    '''
    Load the centroids of all leaf point clouds from the transformation log files
    leaf centroid is the last column of the transformation matrix (3x4) in each file.
    '''
    log_files = os.listdir(directory)
    log_files = sorted(log_files)
    logs = np.asarray([np.loadtxt(os.path.join(directory, leaf)) for leaf in log_files])
    centroids = logs[:,0:3, 3]
    return centroids

def get_dist_mat(data1, data2, training_set=None, mahalanobis_dist=False):
    '''
    Calculate the pairwise distance matrix between two data sets, using Mahalanobis or euclidean distance, without any compression/encoding.
    '''
    if mahalanobis_dist and training_set is not None and training_set.shape[1]>1:
        # need VI, the inverse covariance matrix for Mahalanobis. It is calculated across the entire training set.
        # By default it would be calculated from the inputs, but only works if nr_inputs > nr_features
        Vi = np.linalg.inv(np.cov(training_set.T)).T
        dist =  cdist(data1,data2,'mahalanobis', VI=Vi)
    else:
        # if only 1 feature is given, we use the Euclidean distance. It is directly proportional to the mahalanobis distance for a single variable.
        # same if no training set is given
        print('Using Euclidean distance')
        dist = cdist(data1, data2, 'euclidean')
    return dist

def compute_assignment(dist_mat, label_set_1, label_set_2):
    '''
    Uses the Hungarian method or Munkres algorithm to find the best assignment or minimum weight matching, given a cost matrix.
    '''
    assignment = linear_sum_assignment(dist_mat)
    match = (label_set_1[assignment[0]], label_set_2[assignment[1]])
    return assignment, match, np.array(list(zip(match[0],match[1])))

def get_prediction(data, labels):
    #TODO: Add option to remove emerging leaves from evaluation
    bonn_scores = [[],[],[],[],[],[]] # x, fp, o, dist_x, dist_fp, dist_o
    nr_of_leaves = [] # how many matches were found, dictated by smaller nr of leaves at time t or t+1
    pairings_calculated = [] # number of pairings that were found, regardless of correctness, usually equal to the sum of the min number of leaves at each time step
    total_true_pairings = [] # number of true pairing that could have been found
    bonn_cost_distr=[[],[]]

    for plant in np.unique(labels[:,0]): # for each plant
        subset, sublabels = util.filter_subset(data, labels, plant_nr=plant)
        for time_step in np.unique(sublabels[:,2]): # and for each time step available in the processed data
            if time_step != np.unique(sublabels[:,2])[-1]: # stop at the last timestep, there will be no comparison to make
                c_before, cb_labels = util.filter_subset(subset, sublabels, scan_nr=time_step)
                c_after, ca_labels = util.filter_subset(subset, sublabels, scan_nr=time_step+1)

                # Count how many leaves were present in each time step
                if not nr_of_leaves:
                    nr_of_leaves.append(c_before.shape[0])
                nr_of_leaves.append(c_after.shape[0])

                if c_after.size == 0 or c_before.size == 0:
                    # if any set is empty, there is no comparison to make
                    continue

                centroid_dist = get_dist_mat(c_before, c_after, data, mahalanobis_dist = False)

                c_assignments, temp_match, c_matches = compute_assignment(centroid_dist, cb_labels, ca_labels)

                '''SCORING'''
                true_pairs = np.intersect1d(c_before_labels[:,-1],c_after_labels[:,-1])
                total_true_pairings.append(true_pairs.shape[0])
                pairings_calculated.append(len(c_matches))

                # Regardless of assignments, find the distribution of cost for true vs. false edges in all cost matrices
                true_pair_matrix = cb_labels[:,-1].reshape(-1,1) == ca_labels[:,-1]
                #true_indeces = [np.where(c_before_labels[:,-1]==x)[0] if np.where(c_before_labels[:,-1]==x)[0].size >0 else None for x in c_after_labels[:,-1]]
                bonn_cost_distr[0].append(centroid_dist[np.where(true_pair_matrix==True)])
                bonn_cost_distr[1].append(centroid_dist[np.where(true_pair_matrix==False)])

                if c_matches is not None:
                    x = 0
                    fp = 0
                    o = 0
                    x_dist = []
                    fp_dist = []
                    o_dist = []

##### WORK IN PROGRESS 
                    
                    
                for method, assignment, scores, distances in zip([c_matches, o_matches, a_matches],[c_assignments, o_assignments, a_assignments],[bonn_scores, outline_scores, add_inf_scores],[centroid_dist, outline_dist, add_inf_dist]):
                    if method is not None:
                        x = 0
                        fp = 0
                        o = 0
                        x_dist = []
                        fp_dist = []
                        o_dist = []
                        #scores[3].append(method[:,:,-1])
                        #scores[4].append(np.asarray(assignment))
                        #scores[5].append(distances)
                        for i,pair in enumerate(method):
                            if pair[0,-1] == pair[1,-1]:
                                # A true match is found
                                x += 1
                                x_dist.append(distances[assignment[0][i], assignment[1][i]])
                            elif (pair[0,-1] in true_pairs) ^ (pair[1,-1] in true_pairs):
                                # One leaf from an existing pair was mistakenly matched to an unpaired leaf
                                fp += 1
                                fp_dist.append(distances[assignment[0][i], assignment[1][i]])
                            elif pair[0,-1] not in true_pairs and pair[1,-1] not in true_pairs:
                                o += 1
                                o_dist.append(distances[assignment[0][i], assignment[1][i]])
                        scores[0].append(x)
                        scores[1].append(fp)
                        scores[2].append(o)
                        scores[3].append(x_dist)
                        scores[4].append(fp_dist)
                        scores[5].append(o_dist)

    return bonn_scores, outline_scores, add_inf_scores, nr_of_leaves, pairings_calculated, total_true_pairings, bonn_cost_distr, outline_cost_distr, add_inf_cost_distr

# def analyse(bonn, outline, add_inf, nr_leaves, nr_pairings, nr_true_pairings):
#     #x, fp, o, misses
#     results = []
#     for method in [bonn, outline, add_inf]:
#         if not method[0]:
#             results.append(None)
#         else:
#             x = sum(method[0]) # true positives
#             fp = sum(method[1]) # false positives
#             open = sum(method[2]) # open pairings (type 4 error),
#             misses = sum(nr_true_pairings) - x
#             mean_x_dist = None
#             mean_fp_dist = None
#             mean_open_dist = None
#             average_correct = np.mean(np.asarray(method[0])/np.asarray(nr_true_pairings))

#             if x != 0:
#                 mean_x_dist = sum(sum(ts) for ts in method[3]) / x
#             if fp != 0:
#                 mean_fp_dist = sum(sum(ts) for ts in method[4]) / fp
#             if open != 0:
#                 mean_open_dist = sum(sum(ts) for ts in method[5]) / open

#             method_result = np.array((x, fp, open, misses, mean_x_dist, mean_fp_dist, mean_open_dist, average_correct))
#             results.append(method_result)

#     return results[0], results[1], results[2]