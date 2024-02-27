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
        np.savetxt(os.path.join(os.path.dirname(__file__), 'organ_counts.csv'), all_organ_counts, delimiter=',', fmt='%d')

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
    np.savetxt(os.path.join(os.path.dirname(__file__), 'leaf_areas.csv'), areas, delimiter=',')

    return areas

def biomass(files, voxel_size=3):
    '''
    Given a directory of annotated scans, this function will estimate the biomass of the plant by summing the number of points in the point cloud.
    voxel_size: The size of the voxel grid used to downsample the point cloud, given in mm.
    '''
    class_cats = {"leaf":1, "petiole":2, "berry":3, "flower":4, "crown":5, "background":6, "other":7, "table":8, "emerging leaf":9}
    bio_parts = [1,2,3,4,5,7,9]
    biomass = []
    for scan in files:
        cloud = o3d.geometry.PointCloud()
        data = np.loadtxt(scan, comments="//")
        semantic_labels = data[:,-2]
        for cat in bio_parts:
            cloud.points.extend(o3d.utility.Vector3dVector(data[np.where(semantic_labels==cat)][:,:3]))

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size=voxel_size)
        size = len(voxel_grid.get_voxels()) # number of occupied voxels as a proxy for volume of the plant
        biomass.append(size)
    biomass = np.asarray(biomass)
    np.savetxt(os.path.join(os.path.dirname(__file__), 'plant_biomass.csv'), biomass, delimiter=',')
    return biomass

def biomass_as_func_of_resolution(files):
    class_cats = {"leaf":1, "petiole":2, "berry":3, "flower":4, "crown":5, "background":6, "other":7, "table":8, "emerging leaf":9}
    bio_parts = [1,2,3,4,5,7,9]
    biomass = []
    scan = files[1]
    cloud = o3d.geometry.PointCloud()
    data = np.loadtxt(scan, comments="//")
    semantic_labels = data[:,-2]
    for cat in bio_parts:
        cloud.points.extend(o3d.utility.Vector3dVector(data[np.where(semantic_labels==cat)][:,:3]))

    resolutions = np.arange(0.1, 20, 0.1)
    biomass = []
    for res in resolutions:
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size=res)
        size = len(voxel_grid.get_voxels()) * res**3 # number of occupied voxels times voxel volume
        biomass.append(size)
    biomass = np.asarray(biomass)

    import matplotlib.pyplot as plt
    plt.plot(resolutions, biomass)
    plt.show()
    return biomass