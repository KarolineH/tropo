from minleaf import processing_steps
from minleaf.leaf_op import LeafMinifier
import open3d as o3d

leaf = LeafMinifier(ID='1_0305_0_1', base_dir='/home/karoline/workspace/data/tropo_output/Maize', points=None)

step1 = processing_steps.Alignment(compute=False)
aligned_pc, transform = step1.process(leaf.aligned_file, leaf.transform_log_file, leaf.points)

mesher = processing_steps.Mesher(compute=True)
d_mesh, bp_mesh = mesher.process(None, None, aligned_pc, maxLength=0.75, radii = [0.06,0.1,0.6,1])#radii=[0.1,0.15,0.2,0.25,0.3,0.35,0.4]),0.1,0.5,1

# step3 = processing_steps.Simplifier(compute=True)
# verts, faces, normals, boundary_verts, boundary_edges, full_area = step3.compute(None, None, leaf.bp_mesh_file, closing=30, targetperc=0.1, preserveboundary=True, optimalplacement=True)

# step4 = processing_steps.OutlineFilter(compute=True)
# filtered_verts, filtered_edges,_,_ = step4.process(leaf.filtered_outline_file, boundary_verts, boundary_edges, max_length=0.5)



# draw with open3d
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(aligned_pc)
o3d.visualization.draw_geometries([bp_mesh])
#o3d.visualization.draw_geometries([pc])
print(pc)