# import numpy as np
# from sklearn.metrics import pairwise_distances_argmin
# # create example 3D point cloud
# example_data = np.random.rand(100,3)

# som = SOM(5,5,3)

# class SOM():

#     def __init__(self, m,n,dim):
#         self.m = m
#         self.n = n
#         self.dim = dim
#         self.weights = np.random.rand(m,n,dim)

#     def train(self, point_cloud, num_epochs, eta):

#         for epoch in range(num_epochs):
#             # select a random point from input
#             point = point_cloud[np.random.randint(0,point_cloud.shape[0]),:]
#             # find the best matching unit
#             bmu = pairwise_distances_argmin(point, self.weights, axis=1, metric='euclidean')
#             # update weights
#             self.weights[bmu] += eta * (point - self.weights[bmu])



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
 

class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function and linearly decreasing learning rate.
    Taken from github.com/giannisnik/som
    """
    def __init__(self, m, n, dim, niter, alpha=None, sigma=None):
        super(SOM, self).__init__()
        self.m = 10
        self.n = 10
        self.dim = 3
        self.niter = niter
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        self.weights = torch.randn(m*n, dim)
        self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))
        self.pdist = nn.PairwiseDistance(p=2)

    def get_weights(self):
        return self.weights

    def get_locations(self):
        return self.locations

    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])

    def map_vects(self, input_vects):
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self.weights))],
                            key=lambda x: np.linalg.norm(vect-self.weights[x]))
            to_return.append(self.locations[min_index])

        return to_return

    def forward(self, x, it):
        dists = self.pdist(torch.stack([x for i in range(self.m*self.n)]), self.weights) # distance of each neuron to selected input point
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index,:]
        bmu_loc = bmu_loc.squeeze()
        
        learning_rate_op = 1.0 - it/self.niter # learning rate decreases with iterations
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        bmu_distance_squares = torch.sum(torch.pow(self.locations.float() - torch.stack([bmu_loc for i in range(self.m*self.n)]).float(), 2), 1)
        
        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))
        
        learning_rate_op = alpha_op * neighbourhood_func

        learning_rate_multiplier = torch.stack([learning_rate_op[i:i+1].repeat(self.dim) for i in range(self.m*self.n)])
        delta = torch.mul(learning_rate_multiplier, (torch.stack([x for i in range(self.m*self.n)]) - self.weights))                                         
        new_weights = torch.add(self.weights, delta)
        self.weights = new_weights


m = 10
n = 10
dim = 3
niter = 1000
som = SOM(m, n, dim, niter)
data = torch.randn(100,3)
fig = plt.figure()

for i in range(niter):
    point = data[np.random.randint(0,data.shape[0]),:] # select a random point from the point cloud
    som(point, i) # do a forward pass

    # plot
    if i % 10 == 0:
        # plot vertices and edges using open3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(som.weights)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(som.weights)
        line_set.lines = o3d.utility.Vector2iVector(som.locations)

        # plot neurons individually
        neurons = o3d.geometry.PointCloud()
        neurons.points = o3d.utility.Vector3dVector(data)
        o3d.visualization.draw_geometries([neurons, line_set])


        # plt.title('iteration {}'.format(i))
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # scatterplot = ax.plot(xs=som.weights[:,0], ys=som.weights[:,1], zs=som.weights[:,2], marker='o', ls='')
        # for i, j in som.locations:
        #     ax.plot(som.weights[[i, j], :][:,0], som.weights[[i, j], :][:,1], som.weights[[i, j], :][:,2], color='r', ls='-')
        # plt.show()

weights = som.get_weights() # weights are the final neuron locations in input space
locations = som.get_locations() # locations are the indices of the neurons in the SOM grid
