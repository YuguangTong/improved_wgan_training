import numpy as np
import scipy.misc
import time
import os

image_indices = [73883, 110251, 132301, 57264, 152931, 93861,
                 124938, 79512, 106152, 127384, 134028, 67874,
                 10613, 36510, 198694, 100990]

def make_generator(data_dir, n_files, batch_size):
    epoch_count = [1]

    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        files = [name for name in os.listdir(data_dir)
                 if os.path.isfile(os.path.join(data_dir, name))]
        # remove testset
        for i in image_indices:
            test_file = "{}.jpg".format(str(i).zfill(6))
            files.remove(test_file)
        assert n_files == len(files) + len(image_indices)
        
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, name in enumerate(files):
            image = scipy.misc.imread("{}/{}".format(data_dir, name))
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch


def make_testset(data_dir, image_indices=image_indices):
    images = np.zeros((len(image_indices), 3, 64, 64), dtype=np.int32)
    for n, i in enumerate(image_indices):
        name = "{}.jpg".format(str(i).zfill(6))
        filename = os.path.join(data_dir, name)
        image = scipy.misc.imread(filename)
        images[n] = image.transpose(2, 0, 1)
    return images


def load(batch_size, data_dir='/home/Tong/improved_wgan_training/data/celebA_64x64'):
    if not os.path.isdir(data_dir):
        raise Exception("{} is not a directory".format(data_dir))
    file_count = 202599
    print('load {} files'.format(file_count))
    return make_generator(data_dir, file_count, batch_size), make_testset(data_dir)

if __name__ == '__main__':
    train_gen, test_images  = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print("{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 5:
            print(batch[0].shape, batch[0].dtype)
            break
        t0 = time.time()
