import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Radial Basis Function Kernel
class Kernel:
  
  def __init__(self, x, y) -> None:
    self.x = x
    self.y = y
    self.kernel_matrix = None
    self.alpha = None

  def kernel_distance_metric(self, x1, x2, h):
    """Finds the distance between two local points on the graph
    and applies an L2 norm raised to the negative power of e to be used
    as the kernel method for the kernel matrix.

    Args:
        x1 (tuple or list): first point
        x2 (tuple or list): second point
        h (float): gaussian deviation
    """
    
    distance = math.pow((x1[0] - x2[0]), 2)
    kernel_distance = math.exp(-(distance)/ (2 * (h ** 2)))
    return kernel_distance

  def radial_basis_kernel(self, event):
    """A radial basis kernel which allows for a kernel to be created in infinite dimensions.
    We can use this to find non-linearity in data sets by using the samples of the data (rows of a matrix).
    We also calculate the alpha values (weights) for the predicting function.

    Args:
        event (button): calls the function event
    """
    # Create the Kernel Matrix an NxN matrix
    self.kernel_matrix = np.ones(shape=(self.x.shape[0], self.x.shape[0]), dtype=float)
    for i in range(0, self.x.shape[0]):
      for j in range(0, self.x.shape[0]):
        self.kernel_matrix[i][j] = self.kernel_distance_metric((self.x[i], self.y[i]), (self.x[j], self.y[j]), 1)
        
    print(self.kernel_matrix.shape)
    
    lamb = 0.1 # value to control overfitting and underfitting
    self.alpha = np.linalg.inv(self.kernel_matrix + lamb * np.identity(self.kernel_matrix.shape[0])) @ self.y
    
    
    # fig, ax = plt.subplots()
    # heatmap = ax.imshow(self.kernel_matrix, cmap='viridis', interpolation='nearest')
    
    
    # plt.colorbar(heatmap)
    # plt.xlabel('Data points')
    # plt.ylabel('Data points')
    # plt.title('Kernel Heat Map')
    # plt.show()
    
    return self.kernel_matrix
    
  def predict(self, x_new : list):
    """Predict the y values for the same x values trained for the kernel.

    Args:
        x_new (list): the new input values used to predict the y values

    Returns:
        NDarray: predicted oupout
    """
    y_predict = np.zeros(x_new.shape)

    for index, x_point in enumerate(x_new):
        kernel_values = np.array([self.kernel_distance_metric((x_point, 0), (self.x[i], 0), 1) for i in range(len(self.x))])
        y_predict[index] = np.sum(self.alpha * kernel_values)  # Weighted sum of kernel values

    return y_predict
    
    
  def fit(self):
    
    plt.scatter(self.x, self.y)
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Random Points')
    
    button_ax = plt.axes([0.8, 0,0.1, 0.075])  
    button = Button(button_ax, 'Kernel Method')
    button.on_clicked(self.radial_basis_kernel)

    plt.show()
    

def main():
  
  # Generate random data
  x = np.linspace(-5, 5, 100)
  y = np.power(x, 4) * 5 + np.power(x,3) * 16 + np.power(x, 2) * 3 + 5 + 0.2*np.random.normal(0, 1, x.shape)
  
  radial = Kernel(x , y)
  radial.fit()
  y_predict = radial.predict(x_new=x)
  
  # Plotting the original data
  plt.scatter(x, y, color='blue', label='True Values', alpha=0.5)

  # Plotting the predicted values
  plt.plot(x, y_predict, color='red', label='Predicted Values', linewidth=2)

  # Adding labels and title
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.title('True vs Predicted Values')
  plt.legend()  # Show legend

  # Show the plot
  plt.show()
  


if __name__ == "__main__":
  main()