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

# def get_features(directory):

#     PCAH, test_ds, test_labels = leaf_encoding.get_encoding(train_split=train_split, random_split=random_split, directory=directory, standardise=standardise, location=location, rotation=rotation, scale=scale, as_features=as_features)

#     return features

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
        #print('Using Euclidean distance')
        dist = cdist(data1, data2, 'euclidean')
    return dist

def compute_assignment(dist_mat, label_set_1, label_set_2):
    '''
    Uses the Hungarian method or Munkres algorithm to find the best assignment or minimum weight matching, given a cost matrix.
    '''
    assignment_idxs = np.asarray(linear_sum_assignment(dist_mat)) # Hungarian method, assignment given in indeces
    assignment_labels = (label_set_1[assignment_idxs[0,:]], label_set_2[assignment_idxs[1,:]])
    prediction = np.array(list(zip(assignment_labels[0],assignment_labels[1])))[:,:,-1]
    return assignment_idxs, prediction

def get_single_bonn_prediction(before_centroids, after_centroids, before_labels, after_labels, training_set=None, mahalanobis_dist=False):
    if before_centroids.size == 0 or after_centroids.size == 0:
        print('One of the provided sets is empty.')
        return
    centroid_dist = get_dist_mat(before_centroids, after_centroids, training_set, mahalanobis_dist=mahalanobis_dist)
    assignments, prediction = compute_assignment(centroid_dist, before_labels, after_labels)
    return prediction

def get_bonn_predictions(data, labels):
    nr_of_leaves = []
    predictions = []
    assignments = []
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
                assignment, prediction = compute_assignment(centroid_dist, cb_labels, ca_labels) # the assignment is given in indeces, the prediction in labels (= leaf instance numbers)
                predictions.append(prediction)
                assignments.append(assignment)
    return predictions, assignments, nr_of_leaves

def get_bonn_scores(data, labels):
    #TODO: Add option to remove leaves from evaluation that are only present in the after set

    scores = [[],[],[],[],[],[]] # x, fp, o, dist_x, dist_fp, dist_o
    nr_of_leaves = [] # how many matches were found, dictated by smaller nr of leaves at time t or t+1
    pairings_calculated = [] # number of pairings that were found, regardless of correctness, usually equal to the sum of the min number of leaves at each time step
    total_true_pairings = [] # number of true pairing that could have been found
    cost_distr=[[],[]]

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
                assignments, prediction = compute_assignment(centroid_dist, cb_labels, ca_labels) # the assignment is given in indeces, the prediction in labels (= leaf instance numbers)

                '''SCORING'''
                true_pairs = np.intersect1d(cb_labels[:,-1],ca_labels[:,-1])
                total_true_pairings.append(true_pairs.shape[0])
                pairings_calculated.append(prediction.shape[0])

                # Regardless of assignments, find the distribution of cost for true vs. false edges in all cost matrices
                true_pair_matrix = cb_labels[:,-1].reshape(-1,1) == ca_labels[:,-1]
                cost_distr[0].append(centroid_dist[np.where(true_pair_matrix==True)])
                cost_distr[1].append(centroid_dist[np.where(true_pair_matrix==False)])

                if prediction is not None:
                    print('Prediction:', prediction)
                    # First evaluate the true positives
                    x = np.where(prediction[:,0] == prediction [:,1])[0].shape[0] # true positives
                    x_indeces = np.vstack((assignments[0,prediction[:,0] == prediction [:,1]], assignments[1,prediction[:,0] == prediction [:,1]])) # where those occured in the cost matrix
                    x_dist = centroid_dist[tuple(x_indeces)] # costs of true positives

                    # Then the false positives and open pairings
                    false_indeces = np.vstack((assignments[0,prediction[:,0] != prediction [:,1]], assignments[1,prediction[:,0] != prediction [:,1]]))
                    false_labels = prediction[prediction[:,0] != prediction [:,1]]
                    fp = 0
                    o = 0
                    fp_dist = []
                    o_dist = []
                    for i,error in enumerate(false_indeces.T):
                        if false_labels[i][0] in true_pairs ^ false_labels[i][1] in true_pairs:
                            # This is a FALSE POSITIVE
                            # One leaf that would have had a match was mistakenly matched to an unpaired leaf
                            fp += 1
                            fp_dist.append(centroid_dist[tuple(error)])
                        elif false_labels[i][0] in true_pairs and false_labels[i][1] in true_pairs:
                            # This is also a FALSE POSITIVE
                            # Both leaves from an existing pair were mistakenly matched to each other
                            fp += 1
                            fp_dist.append(centroid_dist[tuple(error)])
                        elif false_labels[i][0] not in true_pairs and false_labels[i][1] not in true_pairs:
                            # This is an OPEN PAIRING
                            # Both leaves had no true pairing, so this assignment should not affect the score
                            o += 1
                            o_dist.append(centroid_dist[tuple(error)])
                        
                    scores[0].append(x)
                    scores[1].append(fp)
                    scores[2].append(o)
                    scores[3].append(x_dist)
                    scores[4].append(fp_dist)
                    scores[5].append(o_dist)

    return scores, nr_of_leaves, pairings_calculated, total_true_pairings, cost_distr


# def get_heiwolt_scores(data, labels):
#     #TODO: Add option to remove leaves from evaluation that are only present in the after set

#     scores = [[],[],[],[],[],[]] # x, fp, o, dist_x, dist_fp, dist_o
#     nr_of_leaves = [] # how many matches were found, dictated by smaller nr of leaves at time t or t+1
#     pairings_calculated = [] # number of pairings that were found, regardless of correctness, usually equal to the sum of the min number of leaves at each time step
#     total_true_pairings = [] # number of true pairing that could have been found
#     cost_distr=[[],[]]

#     for plant in np.unique(labels[:,0]): # for each plant
#         subset, sublabels = util.filter_subset(data, labels, plant_nr=plant)
#         for time_step in np.unique(sublabels[:,2]): # and for each time step available in the processed data
#             if time_step != np.unique(sublabels[:,2])[-1]: # stop at the last timestep, there will be no comparison to make
#                 c_before, cb_labels = util.filter_subset(subset, sublabels, scan_nr=time_step)
#                 c_after, ca_labels = util.filter_subset(subset, sublabels, scan_nr=time_step+1)

#                 # Count how many leaves were present in each time step
#                 if not nr_of_leaves:
#                     nr_of_leaves.append(c_before.shape[0])
#                 nr_of_leaves.append(c_after.shape[0])

#                 if c_after.size == 0 or c_before.size == 0:
#                     # if any set is empty, there is no comparison to make
#                     continue
#                 centroid_dist = get_dist_mat(c_before, c_after, data, mahalanobis_dist = False)
#                 assignments, prediction = compute_assignment(centroid_dist, cb_labels, ca_labels) # the assignment is given in indeces, the prediction in labels (= leaf instance numbers)

#                 '''SCORING'''
#                 true_pairs = np.intersect1d(cb_labels[:,-1],ca_labels[:,-1])
#                 total_true_pairings.append(true_pairs.shape[0])
#                 pairings_calculated.append(prediction.shape[0])

#                 # Regardless of assignments, find the distribution of cost for true vs. false edges in all cost matrices
#                 true_pair_matrix = cb_labels[:,-1].reshape(-1,1) == ca_labels[:,-1]
#                 cost_distr[0].append(centroid_dist[np.where(true_pair_matrix==True)])
#                 cost_distr[1].append(centroid_dist[np.where(true_pair_matrix==False)])

#                 if prediction is not None:
#                     print('Prediction:', prediction)
#                     # First evaluate the true positives
#                     x = np.where(prediction[:,0] == prediction [:,1])[0].shape[0] # true positives
#                     x_indeces = np.vstack((assignments[0,prediction[:,0] == prediction [:,1]], assignments[1,prediction[:,0] == prediction [:,1]])) # where those occured in the cost matrix
#                     x_dist = centroid_dist[tuple(x_indeces)] # costs of true positives

#                     # Then the false positives and open pairings
#                     false_indeces = np.vstack((assignments[0,prediction[:,0] != prediction [:,1]], assignments[1,prediction[:,0] != prediction [:,1]]))
#                     false_labels = prediction[prediction[:,0] != prediction [:,1]]
#                     fp = 0
#                     o = 0
#                     fp_dist = []
#                     o_dist = []
#                     for i,error in enumerate(false_indeces.T):
#                         if false_labels[i][0] in true_pairs ^ false_labels[i][1] in true_pairs:
#                             # This is a FALSE POSITIVE
#                             # One leaf that would have had a match was mistakenly matched to an unpaired leaf
#                             fp += 1
#                             fp_dist.append(centroid_dist[tuple(error)])
#                         elif false_labels[i][0] in true_pairs and false_labels[i][1] in true_pairs:
#                             # This is also a FALSE POSITIVE
#                             # Both leaves from an existing pair were mistakenly matched to each other
#                             fp += 1
#                             fp_dist.append(centroid_dist[tuple(error)])
#                         elif false_labels[i][0] not in true_pairs and false_labels[i][1] not in true_pairs:
#                             # This is an OPEN PAIRING
#                             # Both leaves had no true pairing, so this assignment should not affect the score
#                             o += 1
#                             o_dist.append(centroid_dist[tuple(error)])
                        
#                     scores[0].append(x)
#                     scores[1].append(fp)
#                     scores[2].append(o)
#                     scores[3].append(x_dist)
#                     scores[4].append(fp_dist)
#                     scores[5].append(o_dist)

#     return scores, nr_of_leaves, pairings_calculated, total_true_pairings, cost_distr

def analyse_scores(scores, nr_of_leaves, pairings_calculated, total_true_pairings):
    #scores are x, fp, o, dist_x, dist_fp, dist_o
    Ex = sum(scores[0]) # true positives
    Efp = sum(scores[1]) # false positives
    Eo = sum(scores[2]) # open pairings (type 4 error),
    misses = sum(total_true_pairings) - Ex
    average_overall_correct = Ex/sum(total_true_pairings) # percentage of detected correct pairings across all time steps 
    average_correct_per_step = np.mean(np.asarray(scores[0])/np.asarray(total_true_pairings)) # average correct pairings per time step

    if Ex != 0:
        mean_x_dist = sum(sum(ts) for ts in scores[3]) / Ex # average cost associated with any correct pairing
    if Efp != 0:
        mean_fp_dist = sum(sum(ts) for ts in scores[4]) / Efp # average cost associated with any false positive
    if Eo != 0:
        mean_open_dist = sum(sum(ts) for ts in scores[5]) / Eo # average cost associated with any open pairing

    results = np.array((Ex, Efp, Eo, misses, mean_x_dist, mean_fp_dist, mean_open_dist, average_overall_correct, average_correct_per_step))
    return results