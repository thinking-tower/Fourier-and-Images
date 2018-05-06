import matplotlib
from matplotlib import animation, rc

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from matplotlib.patches import Circle

fig       = plt.figure()
subplot   = fig.add_subplot(1, 1, 1)    # Axes of 1,1 for the first subplot

center_x, center_y = (0.5, 0.5)
plt.plot([center_x], [center_y], marker='o', markersize=10)

shape = Circle((0.5, 0.5), radius = 0.5, edgecolor = 'black',
                    facecolor = 'none')
subplot.add_patch(shape)

plt.axis('tight')
plt.show()



