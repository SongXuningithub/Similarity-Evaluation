import cv2
from torchvision import transforms
import numpy as np

def motion_blur(image, degree=12, angle=45):
  image = np.array(image)
  M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
  motion_blur_kernel = np.diag(np.ones(degree))
  motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
  motion_blur_kernel = motion_blur_kernel / degree
  blurred = cv2.filter2D(image, -1, motion_blur_kernel)
  # convert to uint8
  cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
  blurred = np.array(blurred, dtype=np.uint8)
  return blurred

def get_image_input(imgname, mode):
    if mode!=2:
        DATAPATH = './test_images/'
        filepath = DATAPATH + imgname
        rawimg = cv2.imread(filepath)
        h, w = rawimg.shape[0:2]
        input_size = 448
    if mode == 2:
        img = cv2.imread("./Fidelity/"+imgname)
        img = cv2.resize(img,(448,448))
    elif mode == 3:
        img = cv2.resize(rawimg, (int(w / h * input_size), input_size))
    elif mode == 4:
        img = rawimg[:, w // 10:]
        img = motion_blur(img)
        img = cv2.resize(img, (int(w / h * input_size), input_size))
    elif mode == 5:
        img = rawimg[:, 0:w * 9 // 10]
        img = motion_blur(img)
        img = cv2.resize(img, (int(w / h * input_size), input_size))
    return img

def img2tensor(img):
    aug = transforms.Compose([
        transforms.ToTensor()
    ])
    imgtensor = aug(img)
    imgtensor = imgtensor.unsqueeze(0)
    return imgtensor

if __name__ == '__main__':
    get_image_input('01.png',2)