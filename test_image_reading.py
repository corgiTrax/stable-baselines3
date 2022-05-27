import numpy as np
from PIL import Image

with open('robosuite_image.npy', 'rb') as f:
    a = np.load(f)
    # print(a)
    img = Image.fromarray(a, 'RGB')
    img = img.rotate(180)
    img.show()
    time.sleep(0.01)
    img.close()