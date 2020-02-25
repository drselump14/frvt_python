""" This is main program """
# Author: Philipp Wagner <bytefish@gmx.de>
# Released to public domain under terms of the BSD Simplified license.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the organization nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#   See <http://www.opensource.org/licenses/bsd-license>

# import bob.measure
import os
import sys

import cv2
import face_recognition
import numpy as np

# Threshold
T = float(os.getenv('THRESHOLD', 0.5))


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is
    given.

    Args: path: Path to a folder with subfolders representing the subjects
    (persons).  sz: A tuple with the size Resizes

    Returns: A list [X,y]

            X: The images, which is a Python list of numpy arrays.  y: The
            corresponding labels (the unique number of the subject, person) in
            a Python list.  """
    c = 0
    X1, y1 = [], []
    X2, y2 = [], []
    for dirname, dirnames, _filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            i = 0
            for filename in os.listdir(subject_path):
                try:
                    filepath = os.path.join(subject_path, filename)
                    im = face_recognition.load_image_file(filepath)

                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    img = np.asarray(im, dtype=np.uint8)
                    if i == 0:
                        X1.append(img)
                        y1.append(subdirname)
                    elif i == 1:
                        X2.append(img)
                        y2.append(subdirname)
                    else:
                        break
                    i += 1
                except IOError:
                    print("I/O error")
                except Exception:
                    print(("Unexpected error:", sys.exc_info()[0]))
                    raise
            c = c + 1
    return [X1, y1, X2, y2]


def calc_impostor_attempts(images: np.ndarray, sample_image: np.ndarray):
    #
    # Take one data for zero cost impostor attempts
    #
    sample_image_encoding = face_recognition.face_encodings(sample_image)[0]

    face_encodings = []
    for index, face in enumerate(images):
        face_encoding = face_recognition.face_encodings(face)[0]
        face_encodings.append(face_encoding)
    return face_recognition.face_distance(face_encodings,
                                          sample_image_encoding)


def calc_genuine_attempts(sources, targets):
    distances = []
    for index in range(len(sources)):
        source_face_encoding = face_recognition.face_encodings(
            sources[index])[0]
        target_face_encoding = face_recognition.face_encodings(
            targets[index])[0]
        distance = face_recognition.face_distance([target_face_encoding],
                                                  source_face_encoding)[0]
        distances.append(distance)

    return distances


def classify_as_true(elem):
    return elem < T


def calc_fmr(negatives):
    print('negatives vector:\n', negatives)
    correct_negatives = list(map(classify_as_true, negatives))
    print('comparing with threshold', T, ':\n', correct_negatives)
    FMR = (float(sum(correct_negatives)) / negatives.size)
    print('FMR:', FMR)
    return FMR


def calc_fnmr(positives):
    print('positives vector:\n', positives)
    false_non_matches = list(map(classify_as_true, positives))
    print('comparing with threshold', T, ':\n', false_non_matches)
    FNMR = 1 - (float(sum(false_non_matches)) / len(positives))
    print('FMNR:', FNMR)
    return FNMR


def print_pair(collection, labels):
    for index in range(len(collection)):
        print(labels[index], ':', collection[index])


# main function
def main():
    X1, y1, X2, y2 = read_images("./att-database-of-faces")
    sample_image = X1.pop()
    X2.pop()
    y1.pop()
    y2.pop()
    negatives = calc_impostor_attempts(X1, sample_image)
    fmr = calc_fmr(negatives)
    print_pair(negatives, y1)

    positives = calc_genuine_attempts(X1, X2)
    fnmr = calc_fnmr(positives)
    print_pair(positives, y1)

    print("FNMR: ", fnmr, " @FMR: ", fmr)


if __name__ == "__main__":
    main()
