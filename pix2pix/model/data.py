import collections
import glob
import math
import os
import random

import tensorflow as tf

from arguments import args
from model.utils import rgb_to_lab, preprocess_lab, preprocess

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch, context")

CROP_SIZE = 256


def load_examples():
    if args.input_dir is None or not os.path.exists(args.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(args.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=args.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if args.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1]  # [height, width, channels]
            a_images = preprocess(raw_input[:, :width // 2, :])
            b_images = preprocess(raw_input[:, width // 2:, :])

    if args.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif args.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if args.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)
            r = tf.image.random_flip_up_down(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [args.scale_size, args.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, args.scale_size - CROP_SIZE + 1, seed=seed)),
                         dtype=tf.int32)
        if args.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif args.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    if args.provide_context:
        with tf.name_scope("context_resize"):
            context_images = tf.image.resize_images(inputs, [CROP_SIZE, CROP_SIZE], method=tf.image.ResizeMethod.AREA)

    if args.provide_context:
        paths_batch, inputs_batch, targets_batch, context_batch = tf.train.batch([paths, input_images, target_images, context_images],
                                                              batch_size=args.batch_size)
    else:
        paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images],
                                                                  batch_size=args.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / args.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
        context=context_batch if args.provide_context else None
    )
