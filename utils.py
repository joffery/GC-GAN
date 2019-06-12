import os
import errno
import numpy as np
import scipy
import scipy.misc
import tensorflow as tf
import random
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class CelebA(object):

    def __init__(self):

        self.dataname = "celeba"
        self.image_size = 64
        self.dims = self.image_size*self.image_size

        # 把landmark点作为一个通道加进去
        self.shape = [self.image_size , self.image_size, 4]


    # load celebA dataset
    def load_celebA(self , image_path):

        # get the list of image path
        images_list = read_image_list(image_path)
        # get the data array of image
        return images_list

    @staticmethod
    def getShapeForData(filenames):

        # 原图已经是对齐好的图像了
        array = [get_image(batch_file, 108, is_crop=False, resize_w = 64,
                           is_grayscale=False) for batch_file in filenames]

        sample_images = np.array(array)
        # return sub_image_mean(array , IMG_CHANNEL)
        return sample_images

    @staticmethod
    def getTrainImages(train_list):
        emotion_dict = {'01_01':0, '02_01':0, '03_01':0, '04_01':0, '04_02':0, '01_02':1, '03_02':1,
                        '02_02': 2, '02_03':3, '03_03':4, '04_03':5}

        img_size = 64
        base_dir = '../../ssd'
        ne_img_list = [ tl[0][:-4]+'.png' for tl in train_list]
        ne_lm_list = [tl[0] for tl in train_list]
        emotion_img_list = [tl[1][:-4]+'.png' for tl in train_list]
        emotion_lm_list = [tl[1] for tl in train_list]
        emotion_label = [ emotion_dict[el[4:9]] for el in emotion_lm_list]
        emotion_label = np.asarray(emotion_label, dtype=np.int32)

        # img_dir = 'multipie_128'
        # lm_dir = 'multipie_128_lm'

        img_dir = 'multipie_nearfrontal_faces_original_withMore_lightness_cropped_face'
        lm_dir = 'multipie_nearfrontal_faces_original_withMore_lightness_lm'
        ne_imgs = [get_image(os.path.join(base_dir, img_dir, batch_file), 108, is_crop=False, resize_w = img_size,
                           is_grayscale=False) for batch_file in ne_img_list]
        ne_imgs = np.asarray(ne_imgs)

        emotion_imgs = [get_image(os.path.join(base_dir, img_dir, batch_file), 108, is_crop=False, resize_w = img_size,
                           is_grayscale=False) for batch_file in emotion_img_list]

        emotion_imgs = np.asarray(emotion_imgs)

        emotion_lms = [get_lms(os.path.join(base_dir, lm_dir, batch_file)) for batch_file in emotion_lm_list]

        emotion_lms = np.asarray(emotion_lms)

        ne_lms = [get_lms(os.path.join(base_dir, lm_dir, batch_file)) for batch_file in ne_lm_list]

        return np.concatenate([ne_imgs, emotion_lms], axis=3), emotion_imgs, ne_lms, emotion_label

    @staticmethod
    def getTrainImages_lm_embed(train_list):
        emotion_dict = {'01_01':0, '02_01':0, '03_01':0, '04_01':0, '04_02':0, '01_02':1, '03_02':1,
                        '02_02': 2, '02_03':3, '03_03':4, '04_03':5}
        img_size = 64
        base_dir = '../../ssd'
        ne_img_list = [ tl[0][:-4]+'.png' for tl in train_list]
        ne_lm_list = [tl[0] for tl in train_list]
        emotion_img_list = [tl[1][:-4]+'.png' for tl in train_list]
        emotion_lm_list = [tl[1] for tl in train_list]
        emotion_label = [ emotion_dict[el[4:9]] for el in emotion_lm_list]
        emotion_label = np.asarray(emotion_label, dtype=np.int32)

        # img_dir = 'multipie_128'
        # lm_dir = 'multipie_128_lm'

        img_dir = 'multipie_nearfrontal_faces_original_withMore_lightness_cropped_face'
        lm_dir = 'multipie_nearfrontal_faces_original_withMore_lightness_lm'

        ne_imgs = np.asarray([get_image(os.path.join(base_dir, img_dir, batch_file), 108, is_crop=False, resize_w = img_size,
                           is_grayscale=False) for batch_file in ne_img_list])

        emotion_imgs = np.asarray([get_image(os.path.join(base_dir, img_dir, batch_file), 108, is_crop=False, resize_w = img_size,
                           is_grayscale=False) for batch_file in emotion_img_list])

        emotion_lms = np.asarray([get_raw_lms(os.path.join(base_dir, lm_dir, batch_file)) for batch_file in emotion_lm_list])

        ne_lms = np.asarray([get_raw_lms(os.path.join(base_dir, lm_dir, batch_file)) for batch_file in ne_lm_list])

        emotion_lms_reference, emotion_label_reference = generate_reference(emotion_lms, emotion_label)

        return ne_imgs, emotion_imgs, ne_lms, emotion_lms, emotion_label, emotion_lms_reference, emotion_label_reference

    @staticmethod
    def landmark_transfer_test():
        img_size = 64
        base_dir = ''
        celeba_lm_dir = 'celeba_transfer/guiding_lms'
        lm_list = os.listdir(celeba_lm_dir)
        ne_img_list = [ tl[:-4]+'.png' for tl in lm_list]

        emotion_lm_list = lm_list

        print('lm_list', lm_list)
        print('ne_img_list', ne_img_list)

        img_dir = 'celeba_transfer/input'

        ne_imgs = [get_image(os.path.join(base_dir, img_dir, batch_file), 108, is_crop=False, resize_w = img_size,
                           is_grayscale=False) for batch_file in ne_img_list]
        ne_imgs = np.asarray(ne_imgs)

        ne_imgs = np.repeat(ne_imgs, 4, axis=0)

        emotion_lms = [get_raw_lms(os.path.join(celeba_lm_dir, batch_file)) for batch_file in emotion_lm_list]

        emotion_lms = np.asarray(emotion_lms)

        emotion_lms = np.repeat(emotion_lms, 4, axis=0)

        return ne_imgs, emotion_lms

    @staticmethod
    def landmark_transfer_test2(train_list):
        img_size = 64
        base_dir = '../../ssd'
        lm_dir = 'test_lm_transfer/ck_test'
        lm_list = os.listdir(os.path.join(lm_dir, 'lms'))
        ne_img_list = train_list
        emotion_img_list = [lm[:-4]+'.png' for lm in lm_list]
        emotion_lm_list = lm_list

        print('lm_list', lm_list)
        print('ne_img_list', ne_img_list)
        print('emotion_img_list', emotion_img_list)

        ne_imgs = [get_image(os.path.join(base_dir, 'multipie_nearfrontal_faces_original_withMore_lightness_cropped_face', batch_file), 108, is_crop=False, resize_w = img_size,
                           is_grayscale=False) for batch_file in ne_img_list]
        ne_imgs = np.asarray(ne_imgs)

        emotion_imgs = [get_image(os.path.join(lm_dir, 'imgs', batch_file), 108, is_crop=False, resize_w = img_size,
                           is_grayscale=True) for batch_file in emotion_img_list]

        emotion_imgs = np.asarray(emotion_imgs)

        emotion_lms = [get_lms(os.path.join(lm_dir, 'lms', batch_file)) for batch_file in emotion_lm_list]

        emotion_lms = np.asarray(emotion_lms) + 0.2


        return np.concatenate([ne_imgs, emotion_lms], axis=3), emotion_imgs


    # 想想为啥把它作为静态方法！！！，仅仅是不用加self了么...
    @staticmethod
    def getNextBatch(input_list, batch_num, batch_size=128):
        return input_list[(batch_num) * batch_size: (batch_num + 1) * batch_size]

def generate_reference(emotion_lms, emotion_label):
    length = len(emotion_lms)
    perm = np.arange(length)
    np.random.shuffle(perm)
    emotion_lms_reference = emotion_lms[perm]
    emotion_label_reference = emotion_label[perm]
    return emotion_lms_reference, emotion_label_reference

def get_lms(lm_path):
    img_size = 64
    landmark = np.loadtxt(lm_path)
    lm_img = np.zeros((img_size,img_size))-1
    for lm in landmark:
        x, y = min(lm[0], img_size-1), min(lm[1], img_size-1)
        lm_img[int(max(x,0)), int(max(y,0))]=1
    return lm_img.reshape((img_size, img_size,1))

def get_raw_lms(lm_path):
    img_size = 64
    half_size = (img_size - 1)/2
    landmark = np.reshape(np.loadtxt(lm_path), (-1,))
    landmark[landmark > (img_size-1)] = img_size - 1
    landmark[landmark < 0] = 0
    landmark = (landmark/half_size) - 1
    return landmark + random.gauss(0,1)

def get_image(image_path , image_size , is_crop=True, resize_w = 64 , is_grayscale = False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w)

def transform(image, npx = 64 , is_crop=False, resize_w = 64):

    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w = resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])
    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    # return scipy.misc.imresize(x[40:218-30, 15:178-15],
    #                            [resize_w, resize_w])
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])

# def get_image(image_path, is_grayscale=False):
#     return np.array(inverse_transform(imread(image_path, is_grayscale)))

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image):
    return (image + 1.) / 2.

def read_image_list(category):

    filenames = []
    print("list file")
    list = os.listdir(category)
    for file in list:
        filenames.append(category + "/" + file)
    print("list file ending!")

    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames

def sample_label():

    num = 64
    label_vector = np.zeros((num , 128), dtype=np.float)
    for i in range(0 , num):
        label_vector[i , (i/8)%2] = 1.0
    return label_vector

def log10(x):

  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def getImageData(path, pick_list):
    array = [get_image(os.path.join(path, batch_file), 108, is_crop=False, resize_w = 64,
                           is_grayscale=False) for batch_file in pick_list]

    sample_images = np.array(array)
    # return sub_image_mean(array , IMG_CHANNEL)
    return sample_images

def celeba_label(celeba_path):

    domainA = []
    domainB = []

    img_list = os.listdir(celeba_path)
    print("loading celeba")
    f = open('list_attr_celeba.txt')
    t = f.readline()
    t = f.readline()

    attributes = t.strip("\n").split(" ")
    for i, at in enumerate(attributes):
        print(i, at)

    t = f.readline()
    count = 0
    while t:
        if count > 10000:
            break

        strs = t.split()
        fname = strs[0].split(".")[0]+'.png'

        print(fname)
        att = int(strs[24])   #16: Eyeglasses 40:Young 21:male 5:bald 24：narrow eyes
        if att == -1:
            if fname in img_list:
                domainA.append(fname)
        else:
            if fname in img_list:
                domainB.append(fname)

        count += 1
        t = f.readline()

    f.close()
    return domainA, domainB


