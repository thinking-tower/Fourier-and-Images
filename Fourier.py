import cv2
import numpy as np

from scipy.spatial.distance import cdist

class Image(object):
    def __init__(self, img_loc, shape = None):
        
        # Ensure the image is actually in the current folder!
        self.img = cv2.imread(img_loc)
        if self.img is None:
            print("Image is not in directory.")
        elif shape is not None:
            self.img = cv2.resize(self.img, shape)
            
    def find_path(self):
        
        if self.img is not None:

            # Get the edges of the image
            img = cv2.GaussianBlur(self.img, (3,3), 0)
            edges = cv2.Canny(img, 0, 255)

            ret, thresh = cv2.threshold(edges, 127, 255, 0)
            _, contours, __ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # To simplify things, we only consider the start and end points of the detected contours
            # Note that there are start.shape[0] number of elements that contain (x, y) coordinates
            start, end = np.array([x[0] for x in contours]), np.array([x[-1] for x in contours])
            start = start.reshape(start.shape[0], 2)
            end = end.reshape(end.shape[0], 2)

            # Get the distance between each start and each end point
            distances = cdist(start, end)
            np.fill_diagonal(distances, np.inf)
            self.argsort_col = distances.argsort(axis = 0)
            self.argsort_row = distances.argsort(axis = 1)

            ordered_contours = [contours[idx] for idx in self._sorted_idx()]
            return np.vstack(ordered_contours)

    def _sorted_idx(self, start_point = None, prev_point = None, seen = None, idx_lst = None):
        if start_point is None:
            start_point = 0
            seen = {0}
            idx_lst = [0]

        # Find the indices of points that have:
        # 1) minimum distance from the current point.
        # 2) minimum distance from anywhere else.
        min_dist_arr = self.argsort_col[:, start_point] == self.argsort_row[start_point, :]
        for point in np.argwhere(min_dist_arr == True):
            if point[0] not in seen:
                seen.add(point[0])
                idx_lst += point.tolist() + self._sorted_idx(start_point = point[0], prev_point = start_point, seen = seen, idx_lst = [])

        # If we still have not gone through all the points, go back to the previous point.
        if len(seen) != self.argsort_col.shape[0]:
            idx_lst.append(prev_point)

        return idx_lst

from matplotlib import animation
from matplotlib import pyplot as plt
class Fourier(object):
    def __init__(self, coordinates):
        temp = coordinates[:,:,0] + 1j * coordinates[:,:,1]
        self.complex_coordinates = temp.reshape(temp.shape[0])
        self.N = self.complex_coordinates.size
        self.T = np.arange(self.N)

        self.fig = plt.figure(1)
        self.ax  = self.fig.add_subplot(111)

    def draw(self, n_approximations = 500, speed = 1):
        self._get_circles(n_approximations)
        def update(i):
            for idx, circle in enumerate(self.circle_lst):
                circle.center = (self.locations[idx, i].real, self.locations[idx, i].imag)
            self.final_point.set_data(self.locations[-1, :i].real, self.locations[-1, :i].imag)
            return ([])
        ani = animation.FuncAnimation(self.fig, update, np.arange(0, self.N, speed), interval=1, blit=True)
        lim = 60
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(lim, -lim)
        plt.show()
        
    def _get_circles(self, n_approximations):
        
        # Complex valued fourier series has to be symmetric, so
        # Only frequences multiples from (-n_approximations+1) to 1 and 1 to (n_approximations-1) are considered
        # Which implies we have 2*(n_approximations-1) circles
        circle_locations = np.zeros((2*(n_approximations-1), self.N), dtype = np.complex_)
        self.circle_lst = []
        from itertools import chain
        for idx, multiple in enumerate(chain(range(-n_approximations+1, 0), range(1, n_approximations))):
            fourier_coeff = self._find_fourier_coeff(multiple)
            circle_locations[idx, :] = np.absolute(fourier_coeff) * np.exp(1j * ((2*multiple*np.pi/self.N) * self.T + np.angle(fourier_coeff)))

            circle = plt.Circle((0,0), np.absolute(fourier_coeff), alpha = 1, fill = False)
            self.ax.add_patch(circle)
            self.circle_lst.append(circle)
            
        # Location of each circle's center and the final point
        self.locations = np.add.accumulate(circle_locations, 0)

        # Center circle doesn't move, so remove it!
        self.circle_lst.pop(0)

        # The point that draws the image
        self.final_point, *_ = self.ax.plot(0,0, color='#000000')
    
    def _find_fourier_coeff(self, multiple):
        c = self.complex_coordinates * np.exp(-1j * (2*multiple*np.pi/self.N) * self.T)
        return c.sum() / self.N
        

    
#a = Image("angoo.jpg", (20,20))
#path = a.find_path()
#Fourier(path).draw(1000, speed = 1)
