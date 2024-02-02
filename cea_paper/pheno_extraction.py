import numpy as np
import os
import sys
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from minleaf.leaf_area_zabawa import Surface_Area_Estimation as Zabawa_Area_Estimation

def count_organs(files, save=True):
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

    if save:
        np.savetxt('cea_paper/organ_counts.csv', all_organ_counts, delimiter=',', fmt='%d')

    return all_organ_counts

def leaf_area(directory):
    '''
    (This might take a while)
    Given a directory of annotated scans, this function will estimate the leaf area using the Zabawa method and also from the minimal mesh.
    TODO: Could also use the bp and delauney meshes for comparison.
    '''
    params = {'target_density': 1000,
                'scaling': [0.1,1,5], 
                'hull_param': 0.85,
                'close_holes': False,
                'hole_size':0}

    files = os.listdir(directory + 'processed/aligned/')
    files = sorted(files)
    zmesh_dir = directory + 'processed/zabawa_mesh/'

    areas = []
    for cloud in files:
        aligned_pc = o3d.io.read_point_cloud(directory + 'processed/aligned/' + cloud, format='xyz')
        z_area, mesh = Zabawa_Area_Estimation(np.asarray(aligned_pc.points), target_density=params['target_density'], scaling=params['scaling'], hull_param=params['hull_param'], close_holes=params['close_holes'], hole_size=params['hole_size'])
        o3d.io.write_triangle_mesh(zmesh_dir + cloud.split('.')[0] + '.ply', mesh)
        
        mesh = o3d.io.read_triangle_mesh(directory + 'processed/min_mesh/' + cloud.split('.')[0] + '.obj')
        min_mesh_area = mesh.get_surface_area()

        bp_mesh = o3d.io.read_triangle_mesh(directory + 'processed/bp_meshed/' + cloud.split('.')[0] + '.ply')
        bp_mesh_area = bp_mesh.get_surface_area()

        delauney_mesh = o3d.io.read_triangle_mesh(directory + 'processed/d_meshed/' + cloud.split('.')[0] + '.ply')
        d_mesh_area = delauney_mesh.get_surface_area()

        areas.append([z_area, min_mesh_area, bp_mesh_area, d_mesh_area])
    
    areas = np.asarray(areas)
    np.savetxt('cea_paper/leaf_areas.csv', areas, delimiter=',')

    return areas