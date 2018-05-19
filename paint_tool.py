import numpy as np


class PaintTool:
    def __init__(self, canvas_state, keys, img_size):
        self.keys = keys
        self.canvas_state = canvas_state
        self.mask = np.zeros(img_size)
        self.img_size = img_size

    def paint(self, x, y):
        for i in range(x - 5, x + 5):
            for j in range(y - 5, y + 5):
                i = np.clip(i, 0, self.img_size[1] - 1)
                j = np.clip(j, 0, self.img_size[0] - 1)

                self.mask[j, i] = 1.0

        if self.keys.alt:
            self.canvas_state.remove_mask(self.mask)
        else:
            self.canvas_state.add_mask(self.mask)

    def end(self):
        self.mask[:, :] = 0.0
