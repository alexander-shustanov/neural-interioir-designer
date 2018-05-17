import numpy as np
import tensorflow as tf
import yaml
from skimage.transform import resize

with open("config.yml", 'r') as stream:
    config = yaml.load(stream)


def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))


def gram_matrix(tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[-1])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram


def mask_to_segments(flatten_mask):
    segments = []
    start = None

    for i, val in enumerate(flatten_mask):
        if val > 0.5:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - start))
                start = None

    if start is not None:
        segments.append((start, i + 1 - start))

    return segments


def style_angle(gram1, gram2):
    gram1 = gram1.flatten()
    gram2 = gram2.flatten()
    gram1_length = np.sqrt(np.dot(gram1, gram1))
    gram2_length = np.sqrt(np.dot(gram2, gram2))
    angle_cos = np.dot(gram1, gram2) / gram1_length / gram2_length
    angle_cos = np.clip(angle_cos, 0.0, 1.0)
    return np.arccos(angle_cos)


class Model:
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.load_graph()

    def load_graph(self):
        gd = tf.GraphDef()
        with open("truncated.pb", 'rb') as f:
            gd.ParseFromString(f.read())

        tf.import_graph_def(gd, name="vgg")

        self.input_image = tf.get_default_graph().get_tensor_by_name("vgg/images:0")

        self.layers = ['vgg/conv1_1/conv1_1', 'vgg/conv1_2/conv1_2', 'vgg/conv2_1/conv2_1', 'vgg/conv2_2/conv2_2',
                       'vgg/conv3_1/conv3_1']

    def get_layers(self, indexes):
        ls = []

        for i in indexes:
            ls.append(tf.get_default_graph().get_tensor_by_name(self.layers[i] + ":0"))

        return ls

    def create_feed_dict(self, img):
        return {self.input_image: np.expand_dims(img, axis=0)}

    def create_content_loss(self, content_image, layer_ids):
        feed_dict = self.create_feed_dict(content_image)

        layers = self.get_layers(layer_ids)

        values = self.sess.run(layers, feed_dict=feed_dict)

        layer_losses = []

        for value, layer in zip(values, layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(layer, value_const)
            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses)

        return total_loss

    def create_masked_style_loss(self, style_image, layer_ids, mask):
        style_feed = self.create_feed_dict(style_image)
        mask_feed = self.create_feed_dict(np.stack([mask, mask, mask], axis=2))

        layers = self.get_layers(layer_ids)

        gram_layers = [gram_matrix(layer) for layer in layers]
        values = self.sess.run(gram_layers, feed_dict=style_feed)

        layer_sizes = list(map(lambda l: l.shape[1:3], self.sess.run(layers, feed_dict=mask_feed)))

        layer_masks = [resize(mask, size).reshape([-1]) for size in layer_sizes]
        mask_segments = [mask_to_segments(layer_mask) for layer_mask in layer_masks]

        layer_losses = []

        for style_gram, layer, segments in zip(values, layers, mask_segments):
            style_const = tf.constant(style_gram)

            num_channels = int(layer.get_shape()[-1])

            flatten_layer = tf.reshape(layer, [-1, num_channels])

            parts = [tf.slice(flatten_layer, [start, 0], [size, num_channels]) for (start, size) in segments]

            masked = tf.concat(parts, axis=0)

            loss = mean_squared_error(gram_matrix(masked), style_const)

            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses) / (mask.shape[0] * mask.shape[1] / mask.sum())

        return total_loss

    def create_denoise_loss(self):
        loss = tf.reduce_sum(tf.abs(self.input_image[:, 1:, :, :] - self.input_image[:, :-1, :, :])) + \
               tf.reduce_sum(tf.abs(self.input_image[:, :, 1:, :] - self.input_image[:, :, :-1, :]))

        return loss

    def create_style_signature(self, image, layer_num=0, side=15, stride=2):
        layer = self.get_layers([layer_num])[0]
        layer_gram = gram_matrix(layer)

        height, width, _ = image.shape

        h_s = height // stride
        w_s = width // stride

        signature = np.zeros([h_s, w_s, 64, 64])

        for i in range(h_s):
            for j in range(w_s):
                patch = image[i * stride:i * stride + side, j * stride:j * stride + side, :]
                signature[i, j] = self.sess.run(layer_gram, feed_dict=self.create_feed_dict(patch))

        return signature

    def create_similarity_matrix(self, image=None, signature=None, layer_num=0, side=15, stride=2):
        assert image is not None or signature is not None

        if signature is None:
            signature = self.create_style_signature(image, layer_num=layer_num, side=side, stride=stride)

        h_s, w_s = signature.shape[:2]

        similarity_matrix = np.zeros([h_s - 1, w_s - 1])

        for i in range(h_s - 1):
            for j in range(w_s - 2):
                similarity_matrix[i, j] = \
                    style_angle(signature[i, j], signature[i + 1, j + 1]) + \
                    style_angle(signature[i, j], signature[i, j + 1]) + \
                    style_angle(signature[i, j], signature[i + 1, j])

        similarity_matrix[np.isnan(similarity_matrix)] = 0.0
        similarity_matrix /= 3

        return similarity_matrix
