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

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# tfds works in both Eager and Graph modes
tf.compat.v1.enable_eager_execution()


train_ds = tfds.load("lfw", split="train")
train_ds = train_ds.shuffle(1024).batch(128).repeat(5).prefetch(1)
for example in tfds.as_numpy(train_ds):
    numpy_images, numpy_labels = example["image"], example["label"]
    print(numpy_images)
    print(numpy_labels)
