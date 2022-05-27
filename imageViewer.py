# from skimage.viewer import ImageViewer
# from skimage.io import imread

# img = imread('robosuite_image.png') #path to IMG
# view = ImageViewer(img)
# view.show()

import psutil
for proc in psutil.process_iter():
    print(proc.name())