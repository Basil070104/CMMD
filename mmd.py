from scipy.spatial.distance import cdist
from kernel import Kernel
import numpy as np
import matplotlib.pyplot as plt

class MMD:
  
  def __init__ (self, X, Y, sigma, scale, *args, **kwargs):
    """
    Given two sets of vectors , X = {x1, x2, . . . , xm} and
    Y = {y1, y2, . . . , yn}, sampled from P and Q, respectively, an unbiased estimator for d
    2 MMD(P, Q) is given by, the distance metric
    """  
    self.X = X # samples generated from P
    self.Y = Y # samples generated from Q  
    self.sigma = sigma # kernel std varaince value 
    self.scale = scale
    
  def rbf_kernel(self, X, Y, sigma):
    
    return np.exp(-cdist(X, Y, 'sqeuclidean') / (2 * (sigma**2)))
    
  def compute_mmd(self):
    
    k_xx = self.rbf_kernel(self.X, self.X, self.sigma)
    k_yy = self.rbf_kernel(self.Y, self.Y, self.sigma)
    k_xy = self.rbf_kernel(self.X, self.Y, self.sigma)
    
    m, n = self.X.shape[0], self.Y.shape[0]
    
    # fig, ax = plt.subplots()
    # heatmap = ax.imshow(k_xy, cmap='viridis', interpolation='nearest')
    
    # plt.colorbar(heatmap)
    # plt.xlabel('Data points')
    # plt.ylabel('Data points')
    # plt.title('Kernel Heat Map')
    # plt.show()
    
    return ((1 / (m * (m - 1))) * np.sum(k_xx)) + ((1 / (n * (n - 1))) * np.sum(k_yy))  - ((2 / (m * n)) * np.sum(k_xy))
    # return np.sum(k_xx) + np.sum(k_yy) - (2 * np.sum(k_xy))
    
    
def main():
  # Generate sample data from two different distributions
  X = np.random.normal(0, 1, (100, 2)).reshape(-1, 1)  # Sample from N(0,1)
  # Y = np.random.exponential(5, 1, (100, 2)).reshape(-1, 1)  # Sample from N(1,1)
  Y = np.random.exponential(scale=1.0, size=100).reshape(-1, 1)
  
  print(X.shape)
  
  mmd = MMD(X, Y, 1000, 4)
  distance = mmd.compute_mmd()
  print(f"MMD distance metric of the two density functions : {distance}")
  pass

if __name__ == "__main__":
  main()
    
    
    