import numpy as np
import cv2
import random

# #--- flip ---
def do_random_hflip(image, mask):
    if np.random.rand()>0.5:
        image = cv2.flip(image,1)
        mask = cv2.flip(mask,1)
    return image, mask

def do_random_wflip(image, mask):
    if np.random.rand()>0.5:
        image = cv2.flip(image,0)
        mask = cv2.flip(mask,0)
    return image, mask


# #--- geometric ---
def do_random_rotate(image, mask, mag=15 ):
    angle = np.random.uniform(-1, 1)*mag

    height, width = image.shape[:2]
    cx, cy = width // 2, height // 2

    transform = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image, mask


def do_random_scale( image, mask, mag=0.1 ):
    s = 1 + np.random.uniform(-1, 1)*mag
    height, width = image.shape[:2]
    w,h = int(s*width), int(s*height)
    if (h,w)==image.shape[:2]:
        return image, mask

    dst = np.array([
        [0,0],[width,height], [width,0], #[0,height],
    ]).astype(np.float32)

    if s>1:
        dx = np.random.choice(w-width)
        dy = np.random.choice(h-height)
        src = np.array([
            [-dx,-dy],[-dx+w,-dy+h], [-dx+w,-dy],#[-dx,-dy+h],#
        ]).astype(np.float32)
    if s<1:
        dx = np.random.choice(width-w)
        dy = np.random.choice(height-h)
        src = np.array([
            [dx,dy], [dx+w,dy+h], [dx+w,dy],#
        ]).astype(np.float32)

    transform = cv2.getAffineTransform(src, dst)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine( mask, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def do_random_stretch_y( image, mask, mag=0.25 ):
    s = 1 + np.random.uniform(-1, 1)*mag
    height, width = image.shape[:2]
    h = int(s*height)
    w = width
    if h==height:
        return image, mask

    dst = np.array([
        [0,0],[width,height], [width,0], #[0,height],
    ]).astype(np.float32)


    if s>1:
        dx = 0#np.random.choice(w-width)
        dy = np.random.choice(h-height)
        src = np.array([
            [-dx,-dy],[-dx+w,-dy+h], [-dx+w,-dy],#[-dx,-dy+h],#
        ]).astype(np.float32)
    if s<1:
        dx = 0#np.random.choice(width-w)
        dy = np.random.choice(height-h)
        src = np.array([
            [dx,dy], [dx+w,dy+h], [dx+w,dy],#
        ]).astype(np.float32)

    transform = cv2.getAffineTransform(src, dst)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine( mask, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask



def do_random_stretch_x( image, mask, mag=0.25 ):
    s = 1 + np.random.uniform(-1, 1)*mag
    height, width = image.shape[:2]
    h = height
    w = int(s*width)
    if w==width:
        return image, mask

    dst = np.array([
        [0,0],[width,height], [width,0], #[0,height],
    ]).astype(np.float32)

    if s>1:
        dx = np.random.choice(w-width)
        dy = 0#np.random.choice(h-height)
        src = np.array([
            [-dx,-dy],[-dx+w,-dy+h], [-dx+w,-dy],#[-dx,-dy+h],#
        ]).astype(np.float32)
    if s<1:
        dx = np.random.choice(width-w)
        dy = 0#np.random.choice(height-h)
        src = np.array([
            [dx,dy], [dx+w,dy+h], [dx+w,dy],#
        ]).astype(np.float32)

    transform = cv2.getAffineTransform(src, dst)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine( mask, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def do_random_shift( image, mask, mag=32 ):
    b = mag
    height, width = image.shape[:2]

    image = cv2.copyMakeBorder(image, b,b,b,b, borderType=cv2.BORDER_CONSTANT, value=0)
    mask  = cv2.copyMakeBorder(mask, b,b,b,b, borderType=cv2.BORDER_CONSTANT, value=0)
    x = np.random.randint(0,2*b)
    y = np.random.randint(0,2*b)
    image = image[y:y+height,x:x+width]
    mask = mask[y:y+height,x:x+width]

    return image, mask

###########################################################################################3



# #--- noise ---
def do_random_blurout(image, size=0.20, num_cut=3):
    height, width = image.shape[:2]
    size = int(size*(height+width)/2)
    for t in range(num_cut):
        x = np.random.randint(0,width- size)
        y = np.random.randint(0,height-size)
        x0 = x
        x1 = x+size
        y0 = y
        y1 = y+size
        image[y0:y1,x0:x1]=image[y0:y1,x0:x1].mean()

    return image

def do_random_guassian_blur(image, mag=[0.1, 2.0]):
    sigma = np.random.uniform(mag[0],mag[1])
    image = cv2.GaussianBlur(image, (23, 23), sigma)
    return image

def do_random_noise(image, mag=0.08):
    height, width = image.shape[:2]

    image = image.astype(np.float32)/255
    noise = np.random.uniform(-1,1,size=(height,width))*mag
    image = image+noise

    image = np.clip(image,0,1)
    image = (image*255).astype(np.uint8)
    return image



# # --- intensity ---
def do_random_intensity_shift_contast(image, mag=[0.3,0.2]):
    image = (image).astype(np.float32)/255
    alpha0 = 1 + random.uniform(-1,1)*mag[0]
    alpha1 = random.uniform(-1,1)*mag[1]
    image = (image+alpha1)
    image = np.clip(image,0,1)
    image = image**alpha0
    image = np.clip(image,0,1)
    image = (image*255).astype(np.uint8)
    return image

#https://answers.opencv.org/question/12024/use-of-clahe/)
def do_random_clahe(image, mag=[[2,4],[6,12]]):
    l = np.random.uniform(*mag[0])
    g = np.random.randint(*mag[1])
    clahe = cv2.createCLAHE(clipLimit=l, tileGridSize=(g, g))

    image = clahe.apply(image)
    return image

# https://github.com/facebookresearch/CovidPrognosis/blob/master/covidprognosis/data/transforms.py
def do_histogram_norm(image, mag=[[2,4],[6,12]]):
    num_bin = 255

    histogram, bin = np.histogram( image.flatten(), num_bin, density=True)
    cdf = histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    equalized = np.interp(image.flatten(), bin[:-1], cdf)
    image = equalized.reshape(image.shape)
    return image





