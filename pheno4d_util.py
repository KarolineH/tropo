import os
import numpy as np
import seaborn as sns
import open3d as o3d

def get_file_locations(input_dir):
    '''
    Finds all files in the Pheno4D directory
    Sorts them into lists with and without annotations
    '''

    if input_dir is None:
        input_dir = os.path.join('/home', 'karoline','workspace', 'data', 'Pheno4D')

    plants = os.listdir(input_dir)
    plants.sort()
    maize_plants = [plant for plant in plants if 'Maize' in plant]
    tomato_plants = [plant for plant in plants if 'Tomato' in plant]

    all_locations = []
    all_annotated_locations = []

    counter = 0
    for set in [maize_plants,tomato_plants]:
        for i,plant in enumerate(set):
            records = os.listdir(os.path.join(input_dir, plant))
            records.sort()
            annotated_records = [record for record in records if '_a' in record]

            file_paths = [os.path.join(input_dir, plant, rec) for rec in records]
            annotated_file_paths = [os.path.join(input_dir, plant, rec) for rec in annotated_records]

            all_locations.append(file_paths)
            all_annotated_locations.append(annotated_file_paths)

            counter+= len(file_paths)
    print("found a total of %i point clouds" % counter)
    return all_locations, all_annotated_locations

def open_file(path):
    '''
    Read the point cloud and labels from one txt file
    '''
    print('Opening file %s'%(path))
    file = open(path,"r")
    content = file.read()
    lines = content.split('\n') # split into a list single rows
    if lines[-1] == '': # trim the last row off if it is empty
        lines = lines[:-1]
    raw = lines
    coordinates = np.array([[float(entry)for entry in line.split(' ')[:3]] for line in raw])
    instance_labels = np.array([[float(entry)for entry in line.split(' ')[3:]] for line in raw])
    plant_ID = os.path.basename(path).split('_a.txt')[0]
    return coordinates,instance_labels, plant_ID

def save_as_ply(pointcloud, file):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.io.write_point_cloud(file, pcd)

def draw_cloud(cloud, labels, draw=True, coordinate_frame=False):
    '''
    Visualises a single point cloud
    input: numpy array
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    try:
        pcd = colour_by_labels(pcd, labels)
    except:
        print("Failed to apply colour by labels")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    if coordinate_frame:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=100))
    if draw ==True:
        vis.run()
        #vis.destroy_window()
    return vis

def colour_by_labels(pcd,labels, labelling_method=0):
    colours = np.array(sns.color_palette())
    colour_array = colours[labels[:,labelling_method].astype(int)]
    pcd.colors = o3d.utility.Vector3dVector(colour_array)
    return pcd

def compare_visual(cloud1, labels1, cloud2, labels2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(cloud1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2)

    try:
        pcd1 = colour_by_labels(pcd1, labels1)
    except:
        print("Failed to apply colour by labels")

    try:
        pcd2 = colour_by_labels(pcd2, labels2)
    except:
        print("Failed to apply colour by labels")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name='left', width=600, height=540, left=0, top=0)
    vis.add_geometry(pcd1)
    vis2 = o3d.visualization.VisualizerWithEditing()
    vis2.create_window(window_name='right', width=600, height=540, left=800, top=0)
    vis2.add_geometry(pcd2)

    while True:
        vis.update_geometry(pcd1)
        if not vis.poll_events():
            break
        vis.update_renderer()

        vis2.update_geometry(pcd2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis.destroy_window()
    vis2.destroy_window()

def split_into_organs(points, labels):
    # Splitting the point cloud into sub-clouds for each unique label
    organs = []
    labels = np.reshape(labels, (-1, 1))
    for label in np.unique(labels).astype(int):
        organs.append((points[(labels.astype(int).flatten()==label),:], labels[(labels.astype(int).flatten()==label),:]))
    return organs

def sort_examples(in_data, in_labels):
    # sort
    out_labels = in_labels[np.lexsort((in_labels[:,3], in_labels[:,1],in_labels[:,0])),:]
    out_data = in_data[np.lexsort((in_labels[:,3], in_labels[:,1],in_labels[:,0])),:]
    return out_data, out_labels