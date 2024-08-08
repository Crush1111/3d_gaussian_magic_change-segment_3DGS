import os
import cv2
import numpy as np

PATH = f'/home/guozebin/work_code/f2-nerf/data/reconstruction_xbrain/orbslamsmall/depth'
if __name__ == '__main__':
    count = 0
    # for path_scence in os.listdir(PATH):
    #
    #     path = os.listdir(os.path.join(PATH, path_scence))
    #     path.remove('cameras.json')
    # for img, depth in zip(sorted(os.listdir(PATH), key=lambda x: int(x.split('_')[-1].split('.')[0])),
    #                       sorted(os.listdir(PATH), key=lambda x: int(x.split('_')[-1].split('.')[0]))):

    for img in sorted(os.listdir(PATH), key=lambda x: int(x.split('_')[-1].split('.')[0])):
        print(f'handle {img}')
        count += 1
        # if count <= 300 and count%2 == 0:
        # if count <= 300 and count%2 == 0:
        #     continue
        im = os.path.join(PATH, img)
        img = cv2.imread(im, 0)
        img = ((img / img.max()) * 255).astype(np.uint8)
        cv2.imwrite(im, img)

    print('Done!')
