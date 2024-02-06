import numpy as np
import os
import open3d as o3d
from minleaf.leaf_area_zabawa import Surface_Area_Estimation as Zabawa_Area_Estimation
import minleaf.leaf_angle as incl_angle
import json

'''PARAMETERS'''
data_set = 'Tomato' #'Tomato' #'Maize' #'Strawberry' #'Zombies'
phenotype_out_file = 'phenotypes.json'
mesh_out_dir = 'area_mesh'
hist_out_dir = 'distribution_of_normals'

'''INPUT DATA LOCATIONS'''
data_locations = {'Tomato': '/home/karoline/workspace/data/tropo_output/Tomato',
                'Maize': '/home/karoline/workspace/data/tropo_output/Maize',
                'Strawberry': '/home/karoline/workspace/data/tropo_output/Strawberry',
                'Zombies': '/home/karoline/workspace/data/tropo_output/Zombies'
                }
input_dir = data_locations[data_set]

'''PRE-TUNED PARAMETERS'''
tuned_params = {'Tomato': {'target_density': 400, 
                        'scaling': [0.1,1,10], 
                        'hull_param': 0.85,
                        'close_holes': True,
                        'hole_size':10,
                        'zabawa_mesh': False}, 

                'Maize': {'target_density': 800, 
                        'scaling': [0.1,1,10], 
                        'hull_param': 0.85,
                        'close_holes': True,
                        'hole_size':15,
                        'zabawa_mesh': False},

                'Strawberry': {'target_density': 1000,
                        'scaling': [0.1,1,5], 
                        'hull_param': 0.85,
                        'close_holes': False,
                        'hole_size':0,
                        'zabawa_mesh': False},

                'Zombies': {'target_density': 2000, 
                        'scaling': [0.1,1,5], 
                        'hull_param': 0.85,
                        'close_holes': False,
                        'hole_size':0,
                        'zabawa_mesh': False},
                }
hyperparams = tuned_params[data_set]


def compute_phenotypes(in_dir, hyperparams):

    '''
    Saves all created meshes and individual surface normals to file,
    as well as the computed area and two different computed inclination angles.
    '''

    files = os.listdir(in_dir + '/aligned')
    files = sorted(files)
    phenotypes = {}
    #header = 'plant, date, timestep, leaf, area, main_plane_angle, mormal_x, normal_y, normal_z, mesh_average_angle, avg_norm_x, avg_norm_y, avg_norm_z, centroid_x, centroid_y, centroid_z'

    if not os.path.exists(in_dir + '/' + mesh_out_dir):
        os.makedirs(in_dir + '/' + mesh_out_dir)
    if not os.path.exists(in_dir + '/' + hist_out_dir):
        os.makedirs(in_dir + '/' + hist_out_dir)

    for cloud in files:
        plant, date, timestep, leaf = cloud.split('.')[0].split('_')
        aligned_pc = o3d.io.read_point_cloud(in_dir + '/aligned' + '/' + cloud, format='xyz')
        transform = np.loadtxt(in_dir + '/transform_log' + '/' + cloud)
        centroid = transform[0:3, 3]

        # Compute leaf area
        # sanity check: are all of your leaf point clouds given in the same scale/unit (e.g. mm)? Otherwise area computation is pointless
        if hyperparams['zabawa_mesh']:
            area, mesh = Zabawa_Area_Estimation(np.asarray(aligned_pc.points), target_density=hyperparams['target_density'], scaling=hyperparams['scaling'], hull_param=hyperparams['hull_param'], close_holes=hyperparams['close_holes'], hole_size=hyperparams['hole_size'])
            o3d.io.write_triangle_mesh(in_dir + '/' + mesh_out_dir + '/' + cloud.split('.')[0] + '.ply', mesh)
        else:
            mesh = o3d.io.read_triangle_mesh(in_dir + '/' + 'min_mesh' + '/' + cloud.split('.')[0] + '.obj')
            area = mesh.get_surface_area()

        # Compute inlcination angle
        # main_plane_angle is found by fitting a plane to the leaf cloud
        # mesh_average_angle just takes the average of all mesh surface normals, not weighted or aligned consistently, that normal's angle
        # all_surface_angles lists the inclination angles across all individual mesh surfaces
        # all outputs are given in the global frame
        plane_normal, main_plane_angle = incl_angle.incl_angle_by_plane_fitting(np.asarray(aligned_pc.points), transform)
        mesh_norms, average_norm, mesh_average_angle = incl_angle.incl_angle_by_avg_mesh_norm(mesh, transform)
        all_surface_angles = incl_angle.surface_incl_angle_distr(mesh_norms, transform)

        surface_data = {"normals": mesh_norms.tolist(), "angles": all_surface_angles.tolist()}
        with open(in_dir + '/' + hist_out_dir + '/' + cloud.split('.')[0] + '.json', "w") as json_file:
            json.dump(surface_data, json_file, indent=4)

        # Other phenotypes:
            # orientation angle
            # some shape phenotypes...
            # inclination as a ratio between real and projected leaf area?

        leaf_info = {
            "plant": plant,
            "date": date,
            "timestep": timestep,
            "leaf": leaf,
            "surface_area": area,
            "plane_inclination_angle": main_plane_angle,
            "plane_normal": plane_normal.tolist(),  # Convert to list
            "mesh_average_angle": mesh_average_angle,
            "average_triangle_normal": average_norm.tolist(),  # Convert to list
            "centroid": centroid.tolist()  # Convert to list
            }
        
        phenotypes[cloud.split('.')[0]] = leaf_info

    # Write the data to a JSON file
        filename = in_dir + '/' + phenotype_out_file
        with open(filename, "w") as json_file:
            json.dump(phenotypes, json_file, indent=4)
    return phenotypes

pt = compute_phenotypes(input_dir, hyperparams)