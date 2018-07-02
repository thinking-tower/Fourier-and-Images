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
            img = cv2.GaussianBlur(self.img, (5,5), 0)
            edges = cv2.Canny(img, 100, 255)

            ret, thresh = cv2.threshold(edges, 127, 255, 0)
            _, contours, __ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # To simplify things, we only consider the start and end points of the detected contours
            # Note that there are start.shape[0] number of elements that contain (x, y) coordinates
            # Indexes 0 to self.n_start - 1 correspond to start points
            # Indexes self.n_start onwards correspond to end points
            start, end = np.array([x[0] for x in contours]), np.array([x[-1] for x in contours])
            start_end = np.vstack((start, end))
            self.n_start, self.n_end = start.shape[0], end.shape[0]
            start_end = start_end.reshape(self.n_start+self.n_end , 2)

            # Gets a distance matrix between each start and end point
            # Ignore distance between point's start and end points.
            distances = cdist(start_end, start_end)
            np.fill_diagonal(distances, np.inf)
            np.fill_diagonal(distances[self.n_start:], np.inf)
            np.fill_diagonal(distances[:, self.n_start:], np.inf)

            # start(start end) end(start end)
            distances = np.hstack((distances[:self.n_start], distances[self.n_start:]))
            argsort_row = distances.argsort(axis = 1)[:, :-3]
            args = np.vstack([np.unique(np.mod(argsort_row[i, :], self.n_start), return_index = True)[1] for i in range(self.n_start)])
            argsort_row = argsort_row[np.arange(self.n_start)[:,np.newaxis], args]
            sort_row = distances[np.arange(self.n_start)[:,np.newaxis], argsort_row]

            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import minimum_spanning_tree

            X = csr_matrix(sort_row, dtype= np.int_)
            Tcsr = minimum_spanning_tree(X)

            import networkx as nx
            A = nx.adjacency_matrix(nx.to_networkx_graph(Tcsr)).toarray().astype(np.int_)
            A[A!=0] = argsort_row[np.nonzero(A!=0)]

            def recurs(mst, n, seen, cur = 0, prev = None, pos = None):
                print(cur)
                idx_lst = []
                row = A[cur, :]
                for val in row[row!=0]:
                    if not pos:
                        if val < 2 * n:
                            idx_lst.append((cur, -1 if val < n else 1))
                        else:
                            idx_lst.append((cur, -1 if (val % (2 * n)) < n else 1))
                        pos = idx_lst[-1][1]
                    else:
                        if val < 2 * n:
                            if val < n and pos != -1:
                                idx_lst.append((cur, -1))
                            elif val >= n and pos != 1:
                                idx_lst.append((cur, 1))
                        elif (val % (2*n)) < 2 * n:
                            val = (val % (2*n))
                            if val < n and pos != -1:
                                idx_lst.append((cur, -1))
                            elif val >= n and pos != 1:
                                idx_lst.append((cur, 1))
                        pos = idx_lst[-1][1]
                    if val % n not in seen:
                        seen.add(val%n)
                        idx_lst += recurs(mst, n, seen, val % n, True)
                if prev:
                    idx_lst.append((cur, -pos if pos else -idx_lst[-1][1]))
                return idx_lst
            print(recurs(A, self.n_start, set()), self.n_start)
            ordered_contours = [contours[idx][::stride] for idx, stride in recurs(A, self.n_start, set())]
            return np.vstack(ordered_contours)
        
        
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
 
        self.get_circles(n_approximations)
        
        def update(i):
            for idx, circle in enumerate(self.circle_lst):
                circle.center = (self.locations[idx, i].real, self.locations[idx, i].imag)
            self.final_point.set_data(self.locations[-1, :i].real, self.locations[-1, :i].imag)
            return ([])
        ani = animation.FuncAnimation(self.fig, update, np.arange(0, self.N, speed), interval=1, blit=True)
        lim = np.amax(self.locations[-1, :].real)
        self.ax.set_xlim(np.amin(np.amin(self.locations.real)), np.amax(np.amax(self.locations.real)))
        self.ax.set_ylim(np.amax(np.amax(self.locations.imag)), np.amin(np.amin(self.locations.imag)))
        plt.show()
        
    def get_circles(self, n_approximations):
        
        # Complex valued fourier series has to be symmetric, so
        # Only frequences multiples from (-n_approximations+1) to 1 and 1 to (n_approximations-1) are considered
        # Which implies we have 2*(n_approximations-1) circles
        circle_locations = np.zeros((2*(n_approximations-1), self.N), dtype = np.complex_)
        self.circle_lst = []
        
        from itertools import chain
        for idx, multiple in enumerate(chain(range(-n_approximations+1, 0), range(1, n_approximations))):
            fourier_coeff = self.find_fourier_coeff(multiple)
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
    
    def find_fourier_coeff(self, multiple):
        c = self.complex_coordinates * np.exp(-1j * (2*multiple*np.pi/self.N) * self.T)
        return c.sum() / self.N
        

    
a = Image("gpe.jpg", (200,200))
x = a.find_path()
Fourier(x).draw(2000, speed = 10)
