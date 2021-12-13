import numpy as np
from PIL import Image
from ISR.models import RRDN
img = Image.open('sample_image.jpeg')
#img.resize(size=(img.size[0]*10, img.size[1]*10), resample=Image.BICUBIC)
lr_img = np.array(img)
rrdn = RRDN(weights='gans')
sr_img = rrdn.predict(lr_img)
dest = Image.fromarray(sr_img)
dest.save('result.jpeg')
