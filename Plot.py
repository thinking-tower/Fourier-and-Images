import numpy as np

from matplotlib import animation
from matplotlib import pyplot as plt

# Image.draw
from matplotlib.patches import ConnectionPatch

class Plot(object):
    def __init__(self, period, tup_circles_rad, tup_circles_loc, speed=8, visualize = False):
        self.fig = plt.figure(1)
        self.period = period
        self.tup_circles_loc = tup_circles_loc
        self.speed = speed
        self.visualize = visualize

        # Two circle lists means we have to draw two images and two sets of circles.
        if len(tup_circles_rad) == 2:
            # 224 is the bottom right subplot
            # 222 is the top right subplot
            # 221 is the top left subplot
            # 223 is the bottom left subplot
            self.axes = [self.fig.add_subplot(int(i)) for i in ("224", "222", "221", "223")]
            
            # bottom right/top left subplot = circles, bottom left/top right subplot = images
            # axesA=axes[0], axesB=axes[3] connects bottom right subplot to top right subplot
            # axesA=axes[0], axesB=axes[1] connects bottom right subplot to bottom left subplot
            # axesA=axes[2], axesB=axes[3] connects top left subplot to top right subplot
            # axesA=axes[2], axesB=axes[1] connects top left subplot to bottom left subplot
            self.con_patch_tup = tuple(self.get_con_patch((0,0), (0,0), axesA, axesB) for (axesA, axesB) in zip([0]*2+[2]*2, [1,3]*2))
            self.add_con_patch(self.con_patch_tup)
            self.axes[1].set_zorder(-1)
            self.axes[3].set_zorder(-1)

            # Points that draws the images
            self.final_points = (self.get_final_point(self.axes[1]), self.get_final_point(self.axes[3]))
            self.x_lim = min(np.amin(tup_circles_loc[0][-1].real), np.amin(tup_circles_loc[1][-1].real)), max(np.amax(tup_circles_loc[0][-1].real), np.amax(tup_circles_loc[1][-1].real))
            self.y_lim = max(np.amax(tup_circles_loc[0][-1].imag), np.amax(tup_circles_loc[1][-1].imag)), min(np.amin(tup_circles_loc[0][-1].imag), np.amin(tup_circles_loc[1][-1].imag))

        else:
            self.axes = [self.fig.add_subplot(111)]
            # Point that draws the images
            self.final_points = (self.get_final_point(self.axes[0]),)
            self.x_lim = np.amin(tup_circles_loc[0][-1].real), np.amax(tup_circles_loc[0][-1].real)
            self.y_lim = np.amax(tup_circles_loc[0][-1].imag), np.amin(tup_circles_loc[0][-1].imag)

        if self.visualize is False:
            circle_lst = list()
            axes = (0, 2)
            for n, circle_rad_lst in enumerate(tup_circles_rad):
                circle_lst.append(list())
                for radius in circle_rad_lst:
                    circle = self.get_circle((0,0), radius)
                    self.axes[axes[n]].add_patch(circle)
                    circle_lst[n].append(circle)
                # Center circle doesn't move, so remove it!
                circle_lst[n].pop(0)
            self.tup_circles_lst = tuple(circle_lst)
        
    def get_circle(self, loc, radius):
        return plt.Circle(loc, np.absolute(radius), alpha = 1, fill = False)
    def get_con_patch(self, xyA, xyB, axesA, axesB):
        return ConnectionPatch(xyA=xyA, xyB=xyB,
                               coordsA="data", coordsB="data",
                               axesA=self.axes[axesA], axesB=self.axes[axesB],
                               zorder=25, fc="w", ec="darkblue", lw=2)
    
    def add_con_patch(self, con_patch_tup):
        self.axes[0].add_artist(con_patch_tup[0])
        self.axes[0].add_artist(con_patch_tup[1])
        self.axes[2].add_artist(con_patch_tup[2])
        self.axes[2].add_artist(con_patch_tup[3])

    def get_final_point(self, axis):
        return axis.plot(0,0, color='#000000')[0]
    
    def plot(self, save = False, ani_name = None, ImageMagickLoc = None):
        if self.visualize:
            self.get_visualize()
        else:
            self.get_draw()
        for ax in self.axes:
            ax.set_xlim(self.x_lim)
            ax.set_ylim(self.y_lim)
        ani = animation.FuncAnimation(self.fig, self.update, self.time, interval=1, blit=True)
        if save is True and ImageMagickLoc is not None:
            plt.rcParams['animation.convert_path'] = ImageMagickLoc
            writer = animation.ImageMagickFileWriter(fps = 100)
            ani.save(ani_name if ani_name else 'gif_1.gif', writer=writer)
        else:
            plt.show()

        plt.clf()
        plt.cla()
        plt.close()
        
    def get_draw(self):
        def update(i):
            for n_1, circles_tup in enumerate(self.tup_circles_lst):
                for n_2, circle in enumerate(circles_tup):
                    circle.center = self.get_circle_loc_point(n_1, n_1, circle_idx=n_2, time_idx = i)
            if len(self.tup_circles_lst) == 2:
                self.final_points[0].set_data(self.get_circle_loc_slice(0, 1, -1, i))
                self.final_points[1].set_data(self.get_circle_loc_slice(1, 0, -1, i))
                for con_patch in self.con_patch_tup:
                    con_patch.remove()
                # For readability, not using tuple comprehension
                con_patch_lst = []
                for ((idx_1, idx_2), (idx_3, idx_4)), (axesA, axesB) in zip(zip([(0,0)]*2 + [(1,1)]*2, [(0,1), (1,0)]*2), zip([0]*2+[2]*2, [1,3]*2)):
                    con_patch_lst.append(self.get_con_patch(self.get_circle_loc_point(idx_1, idx_2, -1, i), self.get_circle_loc_point(idx_3, idx_4, -1, i), axesA, axesB))
                self.con_patch_tup = tuple(con_patch_lst)
                self.add_con_patch(self.con_patch_tup)
            else:
                self.final_points[0].set_data(self.get_circle_loc_slice(0, 0, -1, i))
            return ([])
        self.time = np.arange(0, self.period, self.speed)
        self.update = update
        
    def get_circle_loc_point(self, idx_1, idx_2, circle_idx, time_idx):
        return (self.tup_circles_loc[idx_1][circle_idx, time_idx].real, self.tup_circles_loc[idx_2][circle_idx, time_idx].imag)
    
    def get_circle_loc_slice(self, idx_1, idx_2, circle_idx, time_idx):
        return (self.tup_circles_loc[idx_1][circle_idx, :time_idx].real, self.tup_circles_loc[idx_2][circle_idx, :time_idx].imag)

    def get_visualize(self):
        self.n_text = self.axes[0].text(0.02, 0.95, 'Number of Points = 0', transform=self.axes[0].transAxes)
        def update(i):
            self.final_points[0].set_data(self.get_circle_loc_slice(0, 0, i, -1))
            self.n_text.set_text('Number of Fourier Terms = %d' % i)
            return ([])
        self.time = np.arange(0, self.tup_circles_loc[0].shape[0], self.speed)
        self.update = update
