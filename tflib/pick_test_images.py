import os
import scipy.misc
import numpy as np

image_indices = [73883, 110251, 132301, 57264, 152931, 93861,
                 124938, 79512, 106152, 127384, 134028, 67874,
                 10613, 36510, 198694, 100990]

def make_testset(data_dir, image_indices=image_indices):
    images = np.zeros((len(image_indices), 64, 64, 3), dtype=np.float32)
    for n, i in enumerate(image_indices):
        name = "{}.jpg".format(str(i).zfill(6))
        filename = os.path.join(data_dir, name)
        image = scipy.misc.imread(filename)
        images[n] = image
    images = images / 255.
    np.save('testset_label', images)
    return
if __name__ == '__main__':
    make_testset(data_dir='/home/Tong/improved_wgan_training/data/celebA_64x64')
