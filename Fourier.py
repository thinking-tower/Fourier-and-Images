import cv2
import numpy as np
from scipy.spatial.distance import cdist

import cv2
import numpy as np
from scipy.spatial.distance import cdist

import cv2
import numpy as np
from scipy.spatial.distance import cdist

class Image(object):
    def __init__(self, img_loc, shape = None):
        
        # Ensure the image is actually in the current folder!
        self.img = cv2.imread(img_loc)
        if self.img is None:
            print("Image is not in directory.")
        elif shape:
            self.img = cv2.resize(self.img, shape)

    def get_size(self):
        return self.img.shape[0:2]
    
    def find_path(self):
        
        if self.img is not None:

            # Get the edges of the image
            #img = cv2.GaussianBlur(self.img, (5,5), 0)
            edges = cv2.Canny(self.img, 100, 255)

            ret, thresh = cv2.threshold(edges, 127, 255, 0)
            _, contours, __ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            return np.vstack([contours[idx][start::-1] if start and end is None and stride == -1
                                                       else contours[idx] if start is None and end is None and stride == 1
                                                       else contours[idx][:end:-1] if start is None and end and stride == -1
                                                       else contours[idx][start:end:stride] for idx, (start, end, stride) in self.sort(contours)])

    def sort(self, contours):

        order = []

        stack = [(0, 0, None)]
        connections = self.find_connections(contours)

        from bisect import bisect
        while stack:

            start, contour, original = stack.pop()

            if connections[contour]:
                # Go left then right
                pos = bisect([tup[0][0] for tup in connections[contour]], start)
                connection = connections[contour].pop(pos - 1 if pos > 0  else pos)
                order.append((contour, (start, connection[0][0]+1, 1) if connection[0][0]+1 > start else (start, connection[0][0]-1 if connection[0][0] > 0 else 0, -1)))
                stack.append((connection[0][0], contour, original))
                
                if connection[1] in connections:
                    if len(connections[connection[1]]) > 1:
                        stack.append((connection[0][1], connection[1], connection[0][1]))
                    else:
                        stack.append((connection[0][1], connection[1], None))
                else:
                    order.append((connection[1], (connection[0][1], None, -1)))
                    order.append((connection[1], (None, None, 1)))
                    order.append((connection[1], (None, connection[0][1], -1)))
            else:
                if original:
                    order.append((contour, (start, None, 1)))
                    order.append((contour, (None, original - 1 if original else None, -1)))
                else:
                    order.append((contour, (start, None, 1)))
                    order.append((contour, (None, start-1 if start else None, -1)))

        return order

    def find_connections(self, contours):
                                              
        points = np.vstack(contours)
        points = points.reshape((points.shape[0], 2))
        
        dist = cdist(points, points)
        
        len_arr    = np.array([len(contour) for contour in contours], dtype = np.int_)
        end_points = np.add.accumulate(len_arr)

        start     = 0
        start_end = []
        for end in end_points:
            dist[start:end:, start:end:] = np.inf
            start_end.append((start, end))
            start = end
    
        from collections import defaultdict
        connections = defaultdict(list)
        temp_order = [0]
        temp_start_end  = [start_end[0]]
        temp_dist = dist[start_end[0][0]:start_end[0][1]]
        
        # The first connection connects two contours, and the rest connects only one contour
        while len(temp_order) < end_points.size - 1:

            row_min = np.argmin(temp_dist, axis = 0)
            cols = np.indices(row_min.shape)
            col_min = np.argmin(temp_dist[row_min, cols])
            
            temp_row, temp_col = row_min[col_min], col_min
            temp_idx_from = self.find_contour_index(temp_row, temp_start_end)
            idx_from = temp_order[temp_idx_from]
            row = temp_row - temp_start_end[temp_idx_from][0] + start_end[idx_from][0]
            row -= start_end[idx_from][0]
            idx_to = self.find_contour_index(temp_col, start_end)
            col = temp_col - start_end[idx_to][0]

            #print(contours[idx_from][row], contours[idx_to][col])
            
            connections[idx_from].append(((row, col), idx_to))

            for order in temp_order:
                start = start_end[order][0]
                end = start_end[order][1]
                dist[start:end:, start_end[idx_to][0]:start_end[idx_to][1]:] = np.inf
                dist[start_end[idx_to][0]:start_end[idx_to][1]:, start:end:] = np.inf

            temp_order.append(idx_to)

            temp_len_arr    = np.array([len(contours[order]) for order in temp_order], dtype = np.int_)
            temp_end_points = np.add.accumulate(temp_len_arr)

            temp_start     = 0
            temp_start_end = []
            for temp_end in temp_end_points:
                temp_start_end.append((temp_start, temp_end))
                temp_start = temp_end
                
            temp_dist = dist[np.hstack([np.arange(start_end[order][0], start_end[order][1]) for order in temp_order])]

        for idx in connections:
            connections[idx].sort(key = lambda x: x[0][0])

        return connections

    def find_contour_index(self, idx, start_end):
        for i, (start, end) in enumerate(start_end):
            if start <= idx < end:
                return i
        return len(start_end) - 1
        
from matplotlib import animation
from matplotlib import pyplot as plt
class Fourier(object):
    def __init__(self, size, coordinates_1 = None, coordinates_2 = None):

        self.x_lim, self.y_lim = size

        self.n_images = 0
        
        if coordinates_1 is not None:
            
            temp = coordinates_1[:,:,0] + 1j * coordinates_1[:,:,1]
            self.complex_coord_1 = temp.reshape(temp.shape[0])
            self.n_images += 1

        if coordinates_2 is not None:
            
            temp = coordinates_2[:,:,0] + 1j * coordinates_2[:,:,1]
            self.complex_coord_2 = temp.reshape(temp.shape[0])
            self.n_images += 1

            if self.complex_coord_2.size > self.complex_coord_1.size:
                self.complex_coord_1 = np.hstack((self.complex_coord_1, np.zeros((self.complex_coord_2.size - self.complex_coord_1.size), dtype = np.complex_)))
            elif self.complex_coord_1.size > self.complex_coord_2.size:
                self.complex_coord_2 = np.hstack((self.complex_coord_2, np.zeros((self.complex_coord_1.size - self.complex_coord_2.size), dtype = np.complex_)))

    def draw(self, n_approximations = 500, speed = 1, mode = 1):
        
        if self.n_images > 0:
            
            fig = plt.figure(1)
            
            if mode == 1:
                axes = {0: fig.add_subplot(111)} if self.n_images == 1 else {i: fig.add_subplot(int(j)) for i, j in zip(range(4), ("224", "222", "221", "223"))}
            else:
                axes = {i: fig.add_subplot(int(j)) for i, j in zip(range(3), ("224", "222", "221"))}

            final_points, lst_circles_lst, lst_circles_loc, period, largest_radius = self.get_circles(axes, n_approximations, mode = mode)
            
            
            def update(i):
                for n, circles_lst in enumerate(lst_circles_lst):
                    for idx, circle in enumerate(circles_lst):
                        circle.center = (lst_circles_loc[n][idx, i].real, lst_circles_loc[n][idx, i].imag)
                    if mode == 1 and self.n_images == 1:
                        final_points[0].set_data(lst_circles_loc[0][-1, :i].real, lst_circles_loc[0][-1, :i].imag)
                    elif mode == 2 and self.n_images == 1:
                        final_points[0].set_data(lst_circles_loc[0][-1, :i].real, lst_circles_loc[1][-1, :i].imag)
                    elif mode == 1 and self.n_images == 2:
                        final_points[0].set_data(lst_circles_loc[0][-1, :i].real, lst_circles_loc[1][-1, :i].imag)
                        final_points[1].set_data(lst_circles_loc[1][-1, :i].real, lst_circles_loc[0][-1, :i].imag)
                return ([])
            
            ani = animation.FuncAnimation(fig, update, np.arange(0, period, speed), interval=1, blit=True)

            for key, ax in axes.items():
                ax.set_xlim(-(self.x_lim/2 + largest_radius), self.x_lim/2 + largest_radius)
                ax.set_ylim((self.y_lim/2 + largest_radius), -(self.y_lim/2 + largest_radius))
            plt.show()
                        
    def get_circles(self, axes, n_approximations, mode = 1):

        period = self.complex_coord_1.size
        time   = np.arange(period)
        
        largest_radius = -np.inf

        if mode == 1:
        
            circles_lst_1 = []
            circles_loc_1 = np.zeros((2*(n_approximations-1), period), dtype = np.complex_)

            if self.n_images == 2:

                circles_lst_2 = []
                circles_loc_2 = np.zeros((2*(n_approximations-1), period), dtype = np.complex_)
            
            from itertools import chain
            for idx, multiple in enumerate(chain(range(-n_approximations+1, 0), range(1, n_approximations))):
                
                if self.n_images == 1:

                    cn_1 = self.cn(time, period, multiple, self.complex_coord_1)
                            
                    circles_loc_1[idx, :] = self.polar_locations(time, period, multiple, cn_1)
                    circle_1 = plt.Circle((0,0), np.absolute(cn_1), alpha = 1, fill = False)
                    axes[0].add_patch(circle_1)

                elif self.n_images == 2:

                    cn_1 = self.cn(time, period, multiple, self.complex_coord_1.real + 1j * self.complex_coord_2.imag)
                    cn_2 = self.cn(time, period, multiple, self.complex_coord_2.real + 1j * self.complex_coord_1.imag)
                          
                    circles_loc_1[idx, :] = self.polar_locations(time, period, multiple, cn_1)
                    circle_1 = plt.Circle((0,0), np.absolute(cn_1), alpha = 1, fill = False)
                            
                    circles_loc_2[idx, :] = self.polar_locations(time, period, multiple, cn_2)
                    circle_2 = plt.Circle((0,0), np.absolute(cn_2), alpha = 1, fill = False)

                    axes[0].add_patch(circle_1)
                    axes[2].add_patch(circle_2)

                    circles_lst_2.append(circle_2)

                circles_lst_1.append(circle_1)

                largest_radius = max(largest_radius, np.absolute(cn_1), np.absolute(cn_2)) if self.n_images == 2 else max(largest_radius, np.absolute(cn_1))

        elif mode == 2:

            circles_lst_1_cos = []
            circles_lst_1_sine = []
            circles_loc_1 = np.zeros((2*(n_approximations - 1), period), dtype = np.complex_)

            circles_lst_2_cos = []
            circles_lst_2_sine = []
            circles_loc_2 = np.zeros((2*(n_approximations - 1), period), dtype = np.complex_)

            for idx, multiple in enumerate(range(1, n_approximations)):

                an_1 = self.an(time, period, multiple, self.complex_coord_1.real)
                bn_1 = self.bn(time, period, multiple, self.complex_coord_1.real)
                
                an_2 = self.an(time, period, multiple, self.complex_coord_1.imag)
                bn_2 = self.bn(time, period, multiple, self.complex_coord_1.imag)

                circles_loc_1[idx, :] = self.cartesian_locations(time, period, multiple, an_1)
                circles_loc_1[idx + n_approximations - 1, :] = self.cartesian_locations(time, period, multiple, bn_1)
                circles_loc_1[idx + n_approximations - 1, :] = circles_loc_1[idx + n_approximations - 1, :].imag + 1j * circles_loc_1[idx + n_approximations - 1, :].real
                circle_1 = plt.Circle((0,0), an_1, alpha = 1, fill = False)
                circle_2 = plt.Circle((0,0), bn_1, alpha = 1, fill = False)

                circles_loc_2[idx, :] = self.cartesian_locations(time, period, multiple, bn_2)
                circles_loc_2[idx + n_approximations - 1, :] = self.cartesian_locations(time, period, multiple, an_2)
                circles_loc_2[idx + n_approximations - 1, :] = circles_loc_2[idx + n_approximations - 1, :].imag + 1j * circles_loc_2[idx + n_approximations - 1, :].real
                circle_3 = plt.Circle((0,0), bn_2, alpha = 1, fill = False)
                circle_4 = plt.Circle((0,0), an_2, alpha = 1, fill = False)

                axes[0].add_patch(circle_1)
                axes[0].add_patch(circle_2)
                axes[2].add_patch(circle_3)
                axes[2].add_patch(circle_4)
      
                circles_lst_1_cos.append(circle_1)
                circles_lst_1_sine.append(circle_2)
                circles_lst_2_cos.append(circle_3)
                circles_lst_2_sine.append(circle_4)
                
                largest_radius = max(largest_radius, an_1, an_2, bn_1, bn_2)

            circles_lst_1 = circles_lst_1_cos + circles_lst_1_sine
            circles_lst_2 = circles_lst_2_cos + circles_lst_2_sine
            
        # Location of each circle's center and the final point
        circles_loc_1 = np.add.accumulate(circles_loc_1, 0)

        # Center circle doesn't move, so remove it!
        circles_lst_1.pop(0)
                        
        if mode == 1 and self.n_images == 1:

            # The point that draws the image
            final_point_1, *_ = axes[0].plot(0,0, color='#000000')

            return (final_point_1,), (circles_lst_1,), (circles_loc_1,), period, largest_radius

        elif mode == 2 and self.n_images == 1:

            circles_loc_2 = np.add.accumulate(circles_loc_2, 0)
            circles_lst_2.pop(0)
            
            final_point_1, *_ = axes[1].plot(0,0, color='#000000')

            return (final_point_1,), (circles_lst_1, circles_lst_2), (circles_loc_1, circles_loc_2), period, largest_radius
        
        elif mode == 1 and self.n_images == 2:
            
            circles_loc_2 = np.add.accumulate(circles_loc_2, 0)
            circles_lst_2.pop(0)
            
            final_point_1, *_ = axes[1].plot(0,0, color='#000000')
            final_point_2, *_ = axes[3].plot(0,0, color='#000000')

            return (final_point_1, final_point_2), (circles_lst_1, circles_lst_2), (circles_loc_1, circles_loc_2), period, largest_radius


    def an(self, time, period, multiple, coordinates):
        a = coordinates * np.cos(2*multiple*np.pi/period * time)
        return 2*a.sum() / period

    def bn(self, time, period, multiple, coordinates):
        b = coordinates * np.sin(2*multiple*np.pi/period * time)
        return 2*b.sum() / period
    
    def cn(self, time, period, multiple, coordinates):
        c = coordinates * np.exp(-1j * (2*multiple*np.pi/period) * time)
        return c.sum() / period

    def polar_locations(self, time, period, multiple, fourier_coeff):
        return np.absolute(fourier_coeff) * np.exp(1j * ((2*multiple*np.pi/period) * time + np.angle(fourier_coeff)))

    def cartesian_locations(self, time, period, multiple, fourier_coeff):
        return fourier_coeff * np.exp(1j * ((2*multiple*np.pi/period) * time))

    
a = Image("afg.jpg", (200, 200))
b = Image("pikachu.png", (200, 200))
x = a.find_path()
y = b.find_path()
Fourier(a.get_size(), x, y).draw(2000, speed = 1000, mode = 1)
