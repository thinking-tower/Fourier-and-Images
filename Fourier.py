import cv2
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

# Fourier.get_circles
from itertools import chain
# Fourier.draw
from matplotlib.patches import ConnectionPatch
class Fourier(object):
    def __init__(self, coordinates_1 = None, coordinates_2 = None):

        self.fig_n = 1
        self.n_images = 0
        if coordinates_1 is not None:
            temp = coordinates_1[:,:,0] + 1j * coordinates_1[:,:,1]
            self.complex_coord_1 = temp.reshape(temp.shape[0])
            self.n_images += 1
        if coordinates_2 is not None:
            temp = coordinates_2[:,:,0] + 1j * coordinates_2[:,:,1]
            self.complex_coord_2 = temp.reshape(temp.shape[0])
            self.n_images += 1
            # To ensure the two images then have the same "frequency"
            if self.complex_coord_2.size > self.complex_coord_1.size:
                self.complex_coord_1 = np.hstack((self.complex_coord_1, np.full((self.complex_coord_2.size - self.complex_coord_1.size), self.complex_coord_1[-1], dtype = np.complex_)))
            elif self.complex_coord_1.size > self.complex_coord_2.size:
                self.complex_coord_2 = np.hstack((self.complex_coord_2, np.full((self.complex_coord_1.size - self.complex_coord_2.size), self.complex_coord_2[-1], dtype = np.complex_)))

    def visualize(self, save = False, ani_name = None):
        
        fig = plt.figure(self.fig_n)
        self.fig_n += 1
        
        period = self.complex_coord_1.size
        time   = np.arange(period)
        circles_loc_1 = np.zeros((2*(period//2-1), period), dtype = np.complex_)
        circles_rad_1 = np.zeros((2*(period//2-1)), dtype = np.float_)

        for idx, multiple in enumerate(chain(range(-(period//2)+1, 0), range(1, period//2))):
            cn_1 = self.cn(time, period, multiple, self.complex_coord_1)
            circles_rad_1[idx] = np.absolute(cn_1)
            circles_loc_1[idx, :] = self.polar_locations(time, period, multiple, cn_1)
        circles_loc_1 = circles_loc_1[np.argsort(circles_rad_1)[::-1]]  
        circles_loc_1 = np.add.accumulate(circles_loc_1, 0)

        ax = fig.add_subplot(111)
        draw, = ax.plot(0,0)
        n_text = ax.text(0.02, 0.95, 'Number of Points = 0', transform=ax.transAxes)

        def ani(i):
            draw.set_data(circles_loc_1[i, :].real, circles_loc_1[i, :].imag)
            n_text.set_text('Number of Fourier Terms = %d' % i)
            return ([])

        time = np.arange(0, period//2, 8)
        ax.set_xlim(np.amin(circles_loc_1[-1].real), np.amax(circles_loc_1[-1].real))
        ax.set_ylim(np.amax(circles_loc_1[-1].imag), np.amin(circles_loc_1[-1].imag))
        ani = animation.FuncAnimation(fig, ani, time, interval=1, blit=True)
        if save is True:
            plt.rcParams['animation.convert_path'] = 'C:\Program Files\ImageMagick-7.0.8-Q16/magick.exe'
            writer = animation.ImageMagickFileWriter(fps = 100)
            ani.save(ani_name, writer=writer)
        else:
            plt.show()
            
        plt.clf()
        plt.cla()
        plt.close()
    
    def draw(self, n_approximations = 500, speed = 1, mode = 1, save = False , ani_name = None):
        
        fig = plt.figure(1)
        self.fig_n += 1
        # Avoiding aliasing
        n_approximations = self.complex_coord_1.size//2 if n_approximations > self.complex_coord_1.size//2  else n_approximations
            
        if self.n_images == 1 and mode == 1:
            axes = {0: fig.add_subplot(111)}
            final_points, lst_circles_lst, lst_circles_loc = self.get_one_circle_one_image(axes, n_approximations)
            con1 = con2 = con3 = con4 = None
        elif self.n_images == 1 and mode == 2:
            axes = {i: fig.add_subplot(int(j)) for i, j in zip(range(4), ("224", "222", "221", "223"))}
            final_points, lst_circles_lst, lst_circles_loc = self.get_two_circles_one_image(axes, n_approximations)
            con1 = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="data", coordsB="data",
                  axesA=axes[0], axesB=axes[3], zorder=25, fc="w", ec="darkblue", lw=2)
            con2 = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="data", coordsB="data",
                  axesA=axes[0], axesB=axes[1], zorder=25, fc="w", ec="darkblue", lw=2)
            con3 = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="data", coordsB="data",
                  axesA=axes[2], axesB=axes[3], zorder=25, fc="w", ec="darkblue", lw=2)
            con4 = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="data", coordsB="data",
                  axesA=axes[2], axesB=axes[1], zorder=25, fc="w", ec="darkblue", lw=2)

            axes[0].add_artist(con1)
            axes[0].add_artist(con2)
            axes[2].add_artist(con3)
            axes[2].add_artist(con4)
            axes[1].set_zorder(-1)
            axes[3].set_zorder(-1)
            
        else:
            axes = {i: fig.add_subplot(int(j)) for i, j in zip(range(4), ("224", "222", "221", "223"))}
            final_points, lst_circles_lst, lst_circles_loc = self.get_two_circles_two_images(axes, n_approximations)
            con1 = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="data", coordsB="data",
                  axesA=axes[0], axesB=axes[3], zorder=25, fc="w", ec="darkblue", lw=2)
            con2 = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="data", coordsB="data",
                  axesA=axes[0], axesB=axes[1], zorder=25, fc="w", ec="darkblue", lw=2)
            con3 = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="data", coordsB="data",
                  axesA=axes[2], axesB=axes[3], zorder=25, fc="w", ec="darkblue", lw=2)
            con4 = ConnectionPatch(xyA=(0,0), xyB=(0,0), coordsA="data", coordsB="data",
                  axesA=axes[2], axesB=axes[1], zorder=25, fc="w", ec="darkblue", lw=2)

            axes[0].add_artist(con1)
            axes[0].add_artist(con2)
            axes[2].add_artist(con3)
            axes[2].add_artist(con4)
            axes[1].set_zorder(-1)
            axes[3].set_zorder(-1)

        def update(i):
            nonlocal con1, con2, con3, con4
            for n, circles_lst in enumerate(lst_circles_lst):
                for idx, circle in enumerate(circles_lst):
                    circle.center = (lst_circles_loc[n][idx, i].real, lst_circles_loc[n][idx, i].imag)
                if mode == 1 and self.n_images == 1:
                    final_points[0].set_data(lst_circles_loc[0][-1, :i].real, lst_circles_loc[0][-1, :i].imag)
                elif mode == 2 and self.n_images == 1:
                    final_points[0].set_data(lst_circles_loc[0][-1, :i].real, lst_circles_loc[1][-1, :i].imag)
                    final_points[1].set_data(lst_circles_loc[1][-1, :i].real, lst_circles_loc[0][-1, :i].imag)
                    con1.remove()
                    con2.remove()
                    con3.remove()
                    con4.remove()
                    con1 = ConnectionPatch(xyA=(lst_circles_loc[0][-1, i].real, lst_circles_loc[0][-1, i].imag),
                                           xyB=(lst_circles_loc[0][-1, i].real, lst_circles_loc[1][-1, i].imag),
                                           coordsA="data", coordsB="data",
                                           axesA=axes[0], axesB=axes[1], zorder=25, fc="w", ec="darkblue", lw=2)
                    con2 = ConnectionPatch(xyA=(lst_circles_loc[0][-1, i].real, lst_circles_loc[0][-1, i].imag),
                                           xyB=(lst_circles_loc[1][-1, i].real, lst_circles_loc[0][-1, i].imag),
                                           coordsA="data", coordsB="data",
                                           axesA=axes[0], axesB=axes[3], zorder=25, fc="w", ec="darkblue", lw=2)
                    con3 = ConnectionPatch(xyA=(lst_circles_loc[1][-1, i].real, lst_circles_loc[1][-1, i].imag),
                                           xyB=(lst_circles_loc[0][-1, i].real, lst_circles_loc[1][-1, i].imag),
                                           coordsA="data", coordsB="data",
                                           axesA=axes[2], axesB=axes[1], zorder=25, fc="w", ec="darkblue", lw=2)
                    con4 = ConnectionPatch(xyA=(lst_circles_loc[1][-1, i].real, lst_circles_loc[1][-1, i].imag),
                                           xyB=(lst_circles_loc[1][-1, i].real, lst_circles_loc[0][-1, i].imag),
                                           coordsA="data", coordsB="data",
                                           axesA=axes[2], axesB=axes[3], zorder=25, fc="w", ec="darkblue", lw=2)
                    axes[0].add_artist(con1)
                    axes[0].add_artist(con2)
                    axes[2].add_artist(con3)
                    axes[2].add_artist(con4)

                else:
                    final_points[0].set_data(lst_circles_loc[0][-1, :i].real, lst_circles_loc[1][-1, :i].imag)
                    final_points[1].set_data(lst_circles_loc[1][-1, :i].real, lst_circles_loc[0][-1, :i].imag)
                    con1.remove()
                    con2.remove()
                    con3.remove()
                    con4.remove()
                    con1 = ConnectionPatch(xyA=(lst_circles_loc[0][-1, i].real, lst_circles_loc[0][-1, i].imag),
                                           xyB=(lst_circles_loc[0][-1, i].real, lst_circles_loc[1][-1, i].imag),
                                           coordsA="data", coordsB="data",
                                           axesA=axes[0], axesB=axes[1], zorder=25, fc="w", ec="darkblue", lw=2)
                    con2 = ConnectionPatch(xyA=(lst_circles_loc[0][-1, i].real, lst_circles_loc[0][-1, i].imag),
                                           xyB=(lst_circles_loc[1][-1, i].real, lst_circles_loc[0][-1, i].imag),
                                           coordsA="data", coordsB="data",
                                           axesA=axes[0], axesB=axes[3], zorder=25, fc="w", ec="darkblue", lw=2)
                    con3 = ConnectionPatch(xyA=(lst_circles_loc[1][-1, i].real, lst_circles_loc[1][-1, i].imag),
                                           xyB=(lst_circles_loc[0][-1, i].real, lst_circles_loc[1][-1, i].imag),
                                           coordsA="data", coordsB="data",
                                           axesA=axes[2], axesB=axes[1], zorder=25, fc="w", ec="darkblue", lw=2)
                    con4 = ConnectionPatch(xyA=(lst_circles_loc[1][-1, i].real, lst_circles_loc[1][-1, i].imag),
                                           xyB=(lst_circles_loc[1][-1, i].real, lst_circles_loc[0][-1, i].imag),
                                           coordsA="data", coordsB="data",
                                           axesA=axes[2], axesB=axes[3], zorder=25, fc="w", ec="darkblue", lw=2)
                    axes[0].add_artist(con1)
                    axes[0].add_artist(con2)
                    axes[2].add_artist(con3)
                    axes[2].add_artist(con4)
            return ([])

        period = self.complex_coord_1.size
        ani = animation.FuncAnimation(fig, update, np.arange(0, period, speed), interval=1, blit=True)

        if self.n_images == 1 and mode == 1:
            lim_1 = np.amin(lst_circles_loc[0][-1].real), np.amax(lst_circles_loc[0][-1].real)
            lim_2 = np.amax(lst_circles_loc[0][-1].imag), np.amin(lst_circles_loc[0][-1].imag)
        else:
            lim_1 = min(np.amin(lst_circles_loc[0][-1].real), np.amin(lst_circles_loc[1][-1].real)), max(np.amax(lst_circles_loc[0][-1].real), np.amax(lst_circles_loc[1][-1].real))
            lim_2 = max(np.amax(lst_circles_loc[0][-1].imag), np.amax(lst_circles_loc[1][-1].imag)), min(np.amin(lst_circles_loc[0][-1].imag), np.amin(lst_circles_loc[1][-1].imag))
        for key, ax in axes.items():
            ax.set_xlim(lim_1)
            ax.set_ylim(lim_2)

        if save is True:
            plt.rcParams['animation.convert_path'] = 'C:\Program Files\ImageMagick-7.0.8-Q16/magick.exe'
            writer = animation.ImageMagickFileWriter(fps = 100)
            ani.save(ani_name, writer=writer)
        else:
            plt.show()

        plt.clf()
        plt.cla()
        plt.close()
                        
    def get_one_circle_one_image(self, axes, n_approximations):

        period = self.complex_coord_1.size
        time   = np.arange(period)
        circles_lst_1 = []
        circles_loc_1 = np.zeros((2*(n_approximations-1), period), dtype = np.complex_)
        circles_rad_1 = np.zeros((2*(n_approximations-1)), dtype = np.float_)

        for idx, multiple in enumerate(chain(range(-n_approximations+1, 0), range(1, n_approximations))):
            
            cn_1 = self.cn(time, period, multiple, self.complex_coord_1)
            circles_rad_1[idx] = np.absolute(cn_1)
            circles_loc_1[idx, :] = self.polar_locations(time, period, multiple, cn_1)
            circle_1 = plt.Circle((0,0), np.absolute(cn_1), alpha = 1, fill = False)
            axes[0].add_patch(circle_1)
            circles_lst_1.append(circle_1)

        # Sorting big to small
        order = np.argsort(circles_rad_1)[::-1]
        circles_loc_1 = circles_loc_1[order]
        circles_lst_1 = [circles_lst_1[idx] for idx in order]

        # Location of each circle's center and the final point
        circles_loc_1 = np.add.accumulate(circles_loc_1, 0)

        # Center circle doesn't move, so remove it!
        circles_lst_1.pop(0)
                        
        # The point that draws the image
        final_point_1, *_ = axes[0].plot(0,0, color='#000000')

        return (final_point_1,), (circles_lst_1,), (circles_loc_1,)

    def get_two_circles_one_image(self, axes, n_approximations):

        period = self.complex_coord_1.size
        time   = np.arange(period)
        circles_lst_1_cos = []
        circles_lst_1_sine = []
        circles_loc_1 = np.zeros((2*(n_approximations - 1), period), dtype = np.complex_)
        circles_rad_1 = np.zeros((2*(n_approximations - 1)), dtype = np.float_)
        circles_lst_2_cos = []
        circles_lst_2_sine = []
        circles_loc_2 = np.zeros((2*(n_approximations - 1), period), dtype = np.complex_)
        circles_rad_2 = np.zeros((2*(n_approximations - 1)), dtype = np.float_)

        for idx, multiple in enumerate(range(1, n_approximations)):

            cn_1 = self.cn(time, period, multiple, self.complex_coord_1.real)
            an_1, bn_1 = cn_1.real, cn_1.imag

            circles_rad_1[idx] = np.absolute(an_1)
            circles_rad_2[idx+n_approximations-1] = np.absolute(bn_1)
            
            circles_loc_1[idx, :] = self.cartesian_locations(time, period, multiple, an_1)
            circles_loc_1[idx+n_approximations-1, :] = self.cartesian_locations(time, period, multiple, bn_1)
            # A Fourier term is both sine and cos, but self.cartesian_locations(time, period, multiple, bn_1) outputs the sine part as imaginary, so we flip!
            circles_loc_1[idx+n_approximations-1, :] = circles_loc_1[idx + n_approximations - 1, :].imag + 1j * circles_loc_1[idx + n_approximations - 1, :].real
            circle_1 = plt.Circle((0,0), an_1, alpha = 1, fill = False)
            circle_2 = plt.Circle((0,0), bn_1, alpha = 1, fill = False)

            cn_2 = self.cn(time, period, multiple, self.complex_coord_1.imag)
            an_2, bn_2 = cn_2.real, cn_2.imag

            circles_rad_1[idx] = np.absolute(an_2)
            circles_rad_2[idx+n_approximations-1] = np.absolute(bn_2)
            
            circles_loc_2[idx, :] = self.cartesian_locations(time, period, multiple, bn_2)
            circles_loc_2[idx+n_approximations-1, :] = self.cartesian_locations(time, period, multiple, an_2)
                                 
            circles_loc_2[idx+n_approximations-1, :] = circles_loc_2[idx + n_approximations - 1, :].imag + 1j * circles_loc_2[idx + n_approximations - 1, :].real
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

        circles_lst_1 = circles_lst_1_cos + circles_lst_1_sine
        circles_lst_2 = circles_lst_2_cos + circles_lst_2_sine

        
        # Sorting big to small
        order_1 = np.argsort(circles_rad_1)[::-1]
        circles_loc_1 = circles_loc_1[order_1]
        circles_lst_1 = [circles_lst_1[idx] for idx in order_1]
        order_2 = np.argsort(circles_rad_2)[::-1]
        circles_loc_2 = circles_loc_2[order_2]
        circles_lst_2 = [circles_lst_2[idx] for idx in order_2]
        
        # Location of each circle's center and the final point
        circles_loc_1 = np.add.accumulate(circles_loc_1, 0)
        circles_loc_2 = np.add.accumulate(circles_loc_2, 0)

        # Center circle doesn't move, so remove it!
        circles_lst_1.pop(0)
        circles_lst_2.pop(0)
            
        final_point_1, *_ = axes[1].plot(0,0, color='#000000')
        final_point_2, *_ = axes[3].plot(0,0, color='#000000')

        return (final_point_1, final_point_2), (circles_lst_1, circles_lst_2), (circles_loc_1, circles_loc_2)

    def get_two_circles_two_images(self, axes, n_approximations):
        
        period = self.complex_coord_1.size
        time   = np.arange(period)
        circles_lst_1 = []
        circles_loc_1 = np.zeros((2*(n_approximations-1), period), dtype = np.complex_)
        circles_rad_1 = np.zeros((2*(n_approximations-1)), dtype = np.float_)
        circles_lst_2 = []
        circles_loc_2 = np.zeros((2*(n_approximations-1), period), dtype = np.complex_)
        circles_rad_2 = np.zeros((2*(n_approximations-1)), dtype = np.float_)
            
        for idx, multiple in enumerate(chain(range(-n_approximations+1, 0), range(1, n_approximations))):

            cn_1 = self.cn(time, period, multiple, self.complex_coord_1.real + 1j * self.complex_coord_2.imag)
            cn_2 = self.cn(time, period, multiple, self.complex_coord_2.real + 1j * self.complex_coord_1.imag)

            circles_rad_1[idx] = np.absolute(cn_1)
            circles_rad_2[idx] = np.absolute(cn_2)
                  
            circles_loc_1[idx, :] = self.polar_locations(time, period, multiple, cn_1)              
            circles_loc_2[idx, :] = self.polar_locations(time, period, multiple, cn_2)
            circle_1 = plt.Circle((0,0), np.absolute(cn_1), alpha = 1, fill = False)
            circle_2 = plt.Circle((0,0), np.absolute(cn_2), alpha = 1, fill = False)

            axes[0].add_patch(circle_1)
            axes[2].add_patch(circle_2)

            circles_lst_1.append(circle_1)
            circles_lst_2.append(circle_2)
                                 
        # Sorting big to small
        order_1 = np.argsort(circles_rad_1)[::-1]
        circles_loc_1 = circles_loc_1[order_1]
        circles_lst_1 = [circles_lst_1[idx] for idx in order_1]
        order_2 = np.argsort(circles_rad_2)[::-1]
        circles_loc_2 = circles_loc_2[order_2]
        circles_lst_2 = [circles_lst_2[idx] for idx in order_2]
                                 
        # Location of each circle's center and the final point
        circles_loc_1 = np.add.accumulate(circles_loc_1, 0)
        circles_loc_2 = np.add.accumulate(circles_loc_2, 0)

        # Center circle doesn't move, so remove it!
        circles_lst_1.pop(0)
        circles_lst_2.pop(0)
            
        final_point_1, *_ = axes[1].plot(0,0, color='#000000')
        final_point_2, *_ = axes[3].plot(0,0, color='#000000')

        return (final_point_1, final_point_2), (circles_lst_1, circles_lst_2), (circles_loc_1, circles_loc_2)

    def cn(self, time, period, multiple, coordinates):
        c = coordinates * np.exp(-1j * (2*multiple*np.pi/period) * time)
        return c.sum() / period

    def polar_locations(self, time, period, multiple, fourier_coeff):
        return np.absolute(fourier_coeff) * np.exp(1j * ((2*multiple*np.pi/period) * time + np.angle(fourier_coeff)))

    def cartesian_locations(self, time, period, multiple, fourier_coeff):
        return fourier_coeff * np.exp(1j * ((2*multiple*np.pi/period) * time))
