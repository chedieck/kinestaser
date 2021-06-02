import cv2
import variables as V
import os
import json
import numpy as np
from scipy import ndimage
import math
import imageio
import pickle
import random


class LabeledImage():
    def __init__(self, path, A=None, B=None):
        """ A image with two points A and B.

        Parameters
        ----------
        path : str, required
            Path to the image file.
        A : tuple(int, int), optional
            (x, y) coordinates of the first point
        B : tuple(int, int), optional
            (x, y) coordinates of the second point
        """
        self.path = path
        self.filename = path.split('/')[-1]
        self.img = cv2.imread(path)
        self.A = A
        self.B = B

    def show(self, extra=None):
        img_copy = self.img.copy()
        cv2.putText(
            img=img_copy,
            text='A',
            org=self.A,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 0),
            thickness=2,
        )
        cv2.putText(
            img=img_copy,
            text='B',
            org=self.B,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 255),
            thickness=2,
        )
        if extra:
            cv2.putText(
                img=img_copy,
                text=extra,
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 255),
                thickness=2,
            )

        cv2.imshow('image', img_copy)
        cv2.waitKey(0)

    def _get_AB_vector(self):
        return (self.B[0] - self.A[0], -1 * (self.B[1] - self.A[1]))

    def rotate_and_resize(self, angle, multiplier, clockwise=False):
        AB_vector = self._get_AB_vector()
        old_shape = self.img.shape

        # set rotation side
        if clockwise:
            angle = - angle
        cos = math.cos(math.radians(angle))
        sin = math.sin(math.radians(angle))
        rotation_matrix = np.array([[cos, sin],
                                    [-sin, cos]])
        inverted_rotation_matrix = np.array([[cos, -sin],
                                             [sin, cos]])
        new_img = ndimage.rotate(self.img,
                                 angle)  # rotate
        pos_rotation_offset = (int((new_img.shape[1] - old_shape[1]) / 2),
                               int((new_img.shape[0] - old_shape[0]) / 2)) 
        new_shape = (int(new_img.shape[0] * multiplier),
                     int(new_img.shape[1] * multiplier))

        new_img = cv2.resize(new_img, dsize=(new_shape[1],
                                             new_shape[0]))  # resize

        # now we have to recenter the image A
        self.img = new_img
        self.A = (int((self.A[0] + pos_rotation_offset[0]) * multiplier),
                  int((self.A[1] + pos_rotation_offset[1]) * multiplier))
        AB_vector = multiplier * inverted_rotation_matrix.dot(AB_vector)
        self.B = (int(self.A[0] + AB_vector[0]), int(self.A[1] - AB_vector[1]))
    
    def pad_and_center(self, dimensions_dict):
        """Pad or crop an image in the specified dimensions.

        Parameters
        ----------

        dimensions_dict : dict[str: int], required
            Dictionary containing the keys 'left', 'right', 'top' and 'bottom' indicating
            the size of each pad (positive) or crop (negative)
        """

        left = dimensions_dict.get('left')
        right = dimensions_dict.get('right')
        bottom = dimensions_dict.get('bottom')
        top = dimensions_dict.get('top')

        # create empty array of right size
        new_h = self.img.shape[0] + top + bottom
        new_w = self.img.shape[1] + left + right
        new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        # top
        for i in range(top):
            new_img[i] = np.tile(V.CHROMA_KEY, new_w).reshape((new_w, 3))


        # left and right
        for i, row in enumerate(self.img):
            idx = i + top
            left_arr = np.tile(V.CHROMA_KEY, left).reshape((left, 3))
            new_row = np.insert(row, 0, left_arr, axis=0)
            right_arr = np.tile(V.CHROMA_KEY, right).reshape((right, 3))
            new_row = np.append(new_row, right_arr, axis=0)
            new_img[idx] = new_row

        # bottom
        for i in range(bottom):
            idx = i + top + self.img.shape[0]
            new_img[idx] = np.tile(V.CHROMA_KEY, new_w).reshape((new_w, 3))

        # update params
        self.img = new_img
        offset = self._get_AB_vector()
        self.A = (int(self.img.shape[1] / 2), int(self.img.shape[0] / 2))
        self.B = (self.A[0] + offset[0], self.A[1] - offset[1])

    def crop(self, shape, inplace=False):
        """Crop the image.
        """
        if not shape:
            return self.img
        left = int(shape[1] / 2)
        right = shape[1] - left
        top = int(shape[0] / 2)
        bottom = shape[0] - top
        Ax = self.A[0]
        Ay = self.A[1]
        cropped_img = self.img.copy()[Ay - bottom:Ay + top, Ax - left: Ay + right]
        if inplace:
            self.img = cropped_img
            return None
        return cropped_img

    def flip(self, flipcode=1):
        """Should only be used when A is on the center.
        """
        assert self.A[0] == int(self.img.shape[1] / 2) and self.A[1] == int(self.img.shape[0] / 2)
        AB_vector = self._get_AB_vector()
        self.img = cv2.flip(self.img, flipcode)
        self.B = (self.A[0] - AB_vector[0], self.A[1] - AB_vector[1])


class LabeledSequence():
    def __init__(self, limg_list):
        """A sequence of LabeledImage objects.

        Parameters
        ----------

        limg_list : list, required
            List containing the LabeledImage objects.
        """
        self.limg_list = limg_list

        self.min_shape = None
        self.global_A = None
        self.global_B = None
        self.get_canvas_info()

    def show(self):
        for limg in self.limg_list:
            limg.show()

    def save(self, outname='temp'):
        with open(V.LABELED_SEQUENCES_DIR + outname, 'wb') as f:
            pickle.dump(self, f)

    def get_canvas_info(self):
        overflow = {'left': 0,
                    'right': 0,
                    'bottom': 0,
                    'top': 0}

        for limg in self.limg_list:
            shape = limg.img.shape
            A = limg.A
            overflow['left'] = max(overflow['left'], A[0])
            overflow['right'] = max(overflow['right'], shape[1] - A[0])
            overflow['bottom'] = max(overflow['bottom'], shape[0] - A[1])
            overflow['top'] = max(overflow['top'], A[1])

        max_horizontal = max(overflow['right'], overflow['left'])
        max_vertical = max(overflow['top'], overflow['bottom'])
        self.min_shape = (2 * max_vertical, 2 * max_horizontal)
        self.global_A = (max_horizontal, max_vertical)

        # the global B is defined by the first image
        offset = self.limg_list[0]._get_AB_vector()
        self.global_B = (self.global_A[0] + offset[0], self.global_A[1] + offset[1])

    def align_A(self):
        print('MIN_SHAPE:', self.min_shape)
        print('GLOBAL_A:', self.global_A)
        for limg in self.limg_list:
            h, w = limg.img.shape[:2]
            overflow = {'left': self.global_A[0] - limg.A[0],
                        'right': (self.min_shape[1] - self.global_A[0]) - (w - limg.A[0]),
                        'bottom': (self.min_shape[0] - self.global_A[1]) - (h - limg.A[1]),
                        'top': self.global_A[1] - limg.A[1]}

            # center the image in the global A and pad it accordingly
            limg.pad_and_center(overflow)

    def align_B(self, flip=True):
        # we need to recreate the canvas from the paddings created by the rotations

        main_vector = (self.global_B[0] - self.global_A[0], (-1) * (self.global_B[1] - self.global_A[1]))
        for limg in self.limg_list:

            # get the angle between the two vectors
            limg_vector = limg._get_AB_vector()
            dot_product = np.dot(main_vector, limg_vector)
            cosine = dot_product / (np.linalg.norm(main_vector) * np.linalg.norm(limg_vector))
            angle = math.degrees(math.acos(cosine))
            if angle > 90 and flip:
                limg.flip()
                limg_vector = limg._get_AB_vector()
                dot_product = np.dot(main_vector, limg_vector)
                cosine = dot_product / (np.linalg.norm(main_vector) * np.linalg.norm(limg_vector))
                angle = math.degrees(math.acos(cosine))

            # get which side to rotate by
            """
                print(f'before, {clockwise=}')
                print(limg._get_AB_vector())
                limg.show()
                print('after')
                print(limg._get_AB_vector())
                limg.show()
            """
            clockwise=False
            lambda_to_align_vertically = main_vector[0] / limg_vector[0] if limg_vector[0] else 1
            if lambda_to_align_vertically == 0:
                if limg_vector[1] > 0:
                    print('a')
                    if main_vector[0] > 0:
                        print(3)
                        clockwise = True

                elif limg_vector[1] < 0:
                    print('b')
                    if main_vector[0] < 0:
                        print(2)
                        clockwise = True

                else:
                    raise ValueError("A and B can't be the same point")

            else:
                print(f'c {main_vector=}, {limg_vector=} {lambda_to_align_vertically=}')
                if main_vector[1] < (limg_vector[1] * lambda_to_align_vertically):
                    print(1)
                    clockwise = True


            # get multiplier
            multiplier = np.linalg.norm(main_vector) / np.linalg.norm(limg_vector)

            # rotate it and resize so that the B's are also aligned
            limg.rotate_and_resize(angle=angle, multiplier=multiplier, clockwise=clockwise)

    def crop_all(self, shape: tuple, inplace=False):
        cropped_list = []
        for limg in self.limg_list:
            cropped_img = limg.crop(shape, inplace=inplace)
            if not inplace:
                cropped_list.append(cropped_img)
        return cropped_list if cropped_list else None

    def to_gif(self, fps=24, filename='temp.gif', condition=lambda x: True, shape=None):
        img_list = [limg.crop(shape) for limg in self.limg_list if not condition(limg)]
        print(f'making gif with {len(img_list)} / {len(self.limg_list)}')
        imageio.mimsave(V.GIFS_DIR + filename,
                        img_list,
                        fps=fps)

    def align_all(self):
        self.align_A()
        self.align_B()
        self.get_canvas_info()
        self.align_A()

    def shuffle(self):
        random.shuffle(self.limg_list)


def load_labeled_sequence(filename='temp'):
    with open(V.LABELED_SEQUENCES_DIR + filename, 'rb') as f:
        return pickle.load(f)


def select_point(event, x, y, flags, params):
    limg = params.get('limg')
    img = limg.img.copy()
    point = params.get('point')
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.putText(
            img=img,
            text=point,
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2,
        )
        cv2.imshow('image', img)
        if point == 'A':
            limg.A = (x, y)
        if point == 'B':
            limg.B = (x, y)


def label_images_points(folder='tmp/', shuffle=True):
    limg_list = []
    images_path = V.IMAGES_DIR + folder
    folder = [f for f in os.listdir(images_path) if not f.split('.')[-1] in V.SUFFIX_WHITELIST]
    if shuffle:
        random.shuffle(folder)
    try:
        for filename in tqdm(folder):
            limg = LabeledImage(path=images_path + filename)
            print(images_path + filename)
            cv2.imshow('image', limg.img)

            params = {'limg': limg,
                      'point': 'A'}

            cv2.setMouseCallback('image', select_point, params)
            cv2.waitKey(0)

            params['point'] = 'B'
            cv2.imshow('image', limg.img)
            cv2.setMouseCallback('image', select_point, params)
            cv2.waitKey(0)
            limg_list.append(limg)
            cv2.destroyAllWindows()
    except Exception:
        pass
    ret = LabeledSequence(limg_list)
    ret.save()
    return ret


def main(show=False):
    larr = load_labeled_sequence()
    # larr.shuffle()
    square_size = 170
    print("Aligning the A points...")
    larr.align_A()
    print("Aligning the B points...")
    larr.align_B()
    larr.get_canvas_info()
    print("Recentering A points...")
    larr.align_A()
    if show:
        larr.show()
    larr.to_gif(fps=12, filename='12fps_square.gif', condition=conditioner(square_size), shape=[square_size, square_size])

def conditioner(l=180):
    def no_padding_in_square(limg):
        square = limg.crop([l, l])
        return V.CHROMA_KEY in square
    return no_padding_in_square

if __name__ == '__main__':
    main()
    # uncomment the lines below to label the pics on the folder
    # folder = 'sabiá-do-campo/'
    # larr = label_images_points(folder)
