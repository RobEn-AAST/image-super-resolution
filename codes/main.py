import numpy as np
from PIL import Image
from ISR.models import RRDN,RDN


img = Image.open('sample_image.jpeg')
#img.resize(size=(img.size[0]*10, img.size[1]*10), resample=Image.BICUBIC)
lr_img = np.array(img)
rrdn = RRDN(weights='gans')
rdn = RDN(weights='noise-cancel')
sr_img = rrdn.predict(lr_img, by_patch_of_size=50)
sr_img = rdn.predict(sr_img, by_patch_of_size=50)
dest = Image.fromarray(sr_img)
dest.save('result.jpeg')
