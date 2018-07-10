# Fourier and Images

Fourier and Images is a project that tries to draw images with circles.

## Requirements
```
Python 3.x
Matplotlib
OpenCV
Scipy
Numpy
ImageMagick # Only if you want to save to a gif!
```

## Example 

Getting one set of circles with one image:
```
im = Image("pikachu.png", (200, 200))
path = im.sort()
Fourier(path).draw(1000, speed = 8, mode = 1, save = False)
```

[![](https://github.com/thinking-tower/Fourier-and-Images/blob/master/example_gifs/pikachu.gif)](https://github.com/thinking-tower/Fourier-and-Images/blob/master/example_gifs/pikachu.gif "Pikachu")

> Pikachu

Getting two sets of circles with one image:

```
im = Image("einstein.jpg", (200, 200))
path = im.sort()
Fourier(path).draw(1000, speed = 8, mode = 2, save = False)
```

[![](https://github.com/thinking-tower/Fourier-and-Images/blob/master/example_gifs/einstein.gif)](https://github.com/thinking-tower/Fourier-and-Images/blob/master/example_gifs/einstein.gif "Einstein")

> Einstein

Getting two set of circles with two image:
```
im_1 = Image("images/formula.jpeg", (200, 200))
im_2 = Image("images/dickbutt.jpg", (200, 200))
path_1 = im_1.sort()
path_2 = im_2.sort()
# Note setting mode to 2 here doesn't change anything
Fourier(path_1, path_2).draw(1000, speed = 8, mode = 1, save = False)
```

[![](https://github.com/thinking-tower/Fourier-and-Images/blob/master/example_gifs/dickbutt_formula.gif)](https://github.com/thinking-tower/Fourier-and-Images/blob/master/example_gifs/dickbutt_formula.gif "Dickbutt and Euler's Formula")

> Dickbutt and Euler's Formula"

Getting visualization of how number of Fourier Series terms affects the image:
```
im = Image("images/obama.jpg", (200, 200))
path = im.sort()
Fourier(path_5).visualize(save = False, ani_name = "im5.gif")
```

[![](https://github.com/thinking-tower/Fourier-and-Images/blob/master/example_gifs/obama.gif)](https://github.com/thinking-tower/Fourier-and-Images/blob/master/example_gifs/obama.gif "Obama")

> Obama

# Warnings
Too big of an image might cause your computer to freeze! Resizing the image to (200, 200) is a safe choice and anything above (500, 500) starts to get a bit sketchy.

# Animation
Anything above 1000 n_approximations takes a bit of time to animate. Recommend speed = 8.

# Improvements
1) Using FFT to calculate the Fourier Series coefficients
2) Improve edge detection algorithm
3) Improve the function(s) that order the points from the edge detection algorithm

Have fun!
