import numpy as np 
import matplotlib.pyplot as plt
import open3d as o3d
import json

def area_plot(path):
    # Load data
    data = json_dict_to_array(path + '/phenotypes.json')
    ids = data[:,:4].astype(int)

    # Plot leaf area over time 
    # separate figure for each plant, separate line for each leaf 
    for plant in np.unique(ids[:,0]):
        plt.figure(figsize=(10, 6))
        _, plant_ids = filter_ids(ids, plant=plant)
        for leaf in np.unique(plant_ids[:,3]):
            indeces, sample_ids = filter_ids(ids, plant=plant, leaf=leaf)
            timesteps = sample_ids[:,2].astype(float)

            # if 'Tomato' in path:
            #     timesteps[timesteps == 9] = 9.5
            # if 'Maize' in path:
            #     timesteps[timesteps == 5] = 5.5
            leaf_area_data = data[indeces,4]
            plt.plot(timesteps, leaf_area_data, marker='o')
            plt.scatter(timesteps[0], leaf_area_data[0], s=100, marker='o', edgecolors='black', linewidths=3)

        # Format x-axis labels
        dates = np.unique(plant_ids[:,1])
        day = [str(int(date))[-2:] for date in dates]
        month = [str(int(date))[:-2] for date in dates]
        formated_dates = [day + '.' + month for day,month in zip(day,month)]
        # Apply styling
        plt.style.use('seaborn-poster')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Leaf Area [mm2]', fontsize=14)
        plt.title(f'Leaf Area Over Time for Plant {int(plant)}', fontsize=16)
        plt.xticks(ticks = range(0,int(max(plant_ids[:,2])) - int(min(plant_ids[:,2]))+1), labels=formated_dates, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.5)
        plt.show()

def angle_plot(path):
    # Load data
    data = json_dict_to_array(path + '/phenotypes.json')
    ids = data[:,:4].astype(int)

    sample = filter_ids(ids, plant=1, leaf=0)

    # Plot leaf area over time 
    # separate figure for each plant, separate line for each leaf 
    for plant in np.unique(ids[:,0]):
        plt.figure(figsize=(10, 6))
        _, plant_ids = filter_ids(ids, plant=plant)
        for leaf in np.unique(plant_ids[:,3]):
            indeces, sample_ids = filter_ids(ids, plant=plant, leaf=leaf)
            timesteps = sample_ids[:,2].astype(float)

            angle_data = data[indeces,5]

            plt.plot(timesteps, angle_data, marker='o')
            plt.scatter(timesteps[0], angle_data[0], s=100, marker='o', edgecolors='black', linewidths=3)

        # Format x-axis labels
        dates = np.unique(plant_ids[:,1])
        day = [str(int(date))[-2:] for date in dates]
        month = [str(int(date))[:-2] for date in dates]
        formated_dates = [day + '.' + month for day,month in zip(day,month)]
        # Apply styling
        plt.style.use('seaborn-poster')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Leaf Inclination Angle [rad]', fontsize=14)
        plt.title(f'Leaf Inclination Area Over Time for Plant {int(plant)}', fontsize=16)
        plt.xticks(ticks = range(0,int(max(plant_ids[:,2])) - int(min(plant_ids[:,2]))+1), labels=formated_dates, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.5)
        plt.show()

def normal_visualisation(phenotype_path, scan_path):
    # Load data
    data = json_dict_to_array(phenotype_path + '/phenotypes.json')
    ids = data[:,:4].astype(int)
    indeces, sample_ids = filter_ids(ids, plant=1, date=325)
    #scan_file = scan_path + '/Tomato01/T01_0305_a.txt'
    scan_file = scan_path + '/Tomato01/T01_0325_a.txt' 
    #scan_file = scan_path + '/Maize01/M01_0321_a.txt'
    pc = o3d.io.read_point_cloud(scan_file, format='xyz')

    normals = data[indeces,6:9]
    normals2 = data[indeces,10:13]
    centroids = data[indeces,13:]

    points = []
    points = points + list(centroids)
    points = points + (list(centroids + normals*10))
    #points = points + (list(centroids + normals2*10))

    lines = []
    for i in range(len(centroids)):
        lines.append([i, i+len(centroids)])
        #lines.append([i, i+len(centroids)*2])

    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),lines=o3d.utility.Vector2iVector(lines))
    line_set.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pc, line_set])
    return

strawb_id_dict = {'katrina1': 0, 'katrina2': 1, 'katrina3' : 3, 'zara1' : 4, 'zara2': 5, 'zara3' : 6}

def quaternion_from_vectors(v1, v2):
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)

    # Calculate the cross product to find the rotation axis
    a = np.cross(v1_normalized, v2_normalized)
    w = 1 + np.dot(v1_normalized, v2_normalized);
    q = np.append(a, w)
    q = q/np.linalg.norm(q)
    return q


def rotation_matrix_between_vectors(v1, v2):
    # Normalize the input vectors
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)

    # Calculate the cross product to find the rotation axis
    rotation_axis = np.cross(v1_normalized, v2_normalized)

    # Calculate the dot product to find the cosine of the angle
    cos_theta = np.dot(v1_normalized, v2_normalized)

    # Calculate the sine of the angle
    sin_theta = np.linalg.norm(rotation_axis)

    # Construct the rotation matrix using the Rodrigues' rotation formula
    if sin_theta != 0:
        rotation_matrix = (
            np.eye(3) +
            np.sin(np.arccos(cos_theta)) * 
            (rotation_axis / sin_theta) +
            (1 - np.cos(np.arccos(cos_theta))) * 
            np.outer(rotation_axis, rotation_axis) / sin_theta**2
        )
    else:
        # Handle the case where the vectors are collinear
        rotation_matrix = np.eye(3)

    return rotation_matrix

def filter_ids(ids, plant=None, date=None, timestep=None, leaf=None):
    indeces = np.arange(len(ids))
    if plant is not None:
        indeces = indeces[ids[:,0]==plant]
        ids = ids[ids[:,0]==plant]
    if date is not None:
        indeces = indeces[ids[:,1]==date]
        ids = ids[ids[:,1]==date]
    if timestep is not None:
        indeces = indeces[ids[:,2]==timestep]
        ids = ids[ids[:,2]==timestep]
    if leaf is not None:
        indeces = indeces[ids[:,3]==leaf]
        ids = ids[ids[:,3]==leaf]
    return indeces, ids


def json_dict_to_array(path):
    with open(path) as f:
        data = json.load(f)

    # Initialize an empty list to store the rows
    rows = []
    # Iterate through the JSON data
    for key, values in data.items():
        row = []

        # Add the values in the desired order
        if 'Strawberry' in path:
            values['plant'] = strawb_id_dict[values['plant']]
            values['date'] = values['date'][4:]

        row.append(float(values['plant']))
        row.append(float(values['date']))
        row.append(float(values['timestep']))
        row.append(float(values['leaf']))
        row.append(float(values['surface_area']))
        row.append(float(values['plane_inclination_angle']))
        row.extend([float(x) for x in values['plane_normal']])
        row.append(float(values['mesh_average_angle']))
        row.extend([float(x) for x in values['average_triangle_normal']])
        row.extend([float(x) for x in values['centroid']])

        # Append the row to the list
        rows.append(row)

    # Convert the list of rows into a NumPy array
    data_array = np.array(rows)
    return data_array

if __name__=='__main__':
    path = '/home/karoline/workspace/data/tropo_output/Tomato'
    scan_path = '/home/karoline/workspace/data/Pheno4D'
    area_plot(path)
    angle_plot(path)
    #normal_visualisation(path, scan_path)
