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

import bob.measure
import face_recognition
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()

# Threshold
T = 0.5


#
# @params [np.ndarray]
# @return [np.ndarray] array of distances from impostor attempts
#
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


# main function
def main():
    train_ds, info = tfds.load("lfw", split="train", with_info=True)
    train_ds = train_ds.shuffle(1024).batch(128).repeat(5).prefetch(10).take(1)
    images_with_label = next(tfds.as_numpy(train_ds))
    images = images_with_label["image"]
    sample_image = images[0]
    images = np.delete(images, 0, 0)
    negatives = calc_impostor_attempts(images, sample_image)
    print(negatives)

    correct_negatives = bob.measure.correctly_classified_negatives(
        negatives, T)
    FPR = (float(correct_negatives.sum()) / negatives.size)
    print('FPR:', FPR)


# for example in images:
#     numpy_images, numpy_labels = example["image"], example["label"]

if __name__ == "__main__":
    main()
