import numpy as np
import tensorflow as tf

weight_content = 1.5
weight_style = 10.0
weight_denoise = 0.3
num_iterations = 60
step_size = 4.0


class StyleHelper:
    def __init__(self, model, content_image, style_image, mask):
        self.model = model

        loss_content = self.model.create_content_loss(content_image, [4])
        loss_style = self.model.create_masked_style_loss(style_image, [2, 3, 4], mask)
        loss_denoise = self.model.create_denoise_loss()

        adj_content = tf.Variable(1e-10, name='adj_content')
        adj_style = tf.Variable(1e-10, name='adj_style')
        adj_denoise = tf.Variable(1e-10, name='adj_denoise')

        model.sess.run([adj_content.initializer,
                        adj_style.initializer,
                        adj_denoise.initializer])

        update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
        update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
        update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

        loss_combined = weight_content * adj_content * loss_content + \
                        weight_style * adj_style * loss_style + \
                        weight_denoise * adj_denoise * loss_denoise

        gradient = tf.gradients(loss_combined, model.input_image)

        self.run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

    def step(self, img):
        feed_dict = self.model.create_feed_dict(img)

        grad, adj_content_val, adj_style_val, adj_denoise_val = self.model.sess.run(self.run_list, feed_dict=feed_dict)

        grad = np.squeeze(grad)

        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        return grad * step_size_scaled
