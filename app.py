import tkinter as tk
import tkinter.filedialog as fdialog

import numpy as np
from PIL import Image, ImageTk
from skimage.io import imread, imsave
from skimage.measure import label
from skimage.transform import resize

from nn import Model, style_angle

BRUSH = "brush"
MAGIC = "magic"
STYLE = "style"

IMG_SIZE = (300, 300)

TOOLS = [BRUSH, MAGIC, STYLE]


class App:
    def __init__(self):
        self.model = Model()

        self.root = tk.Tk()

        self.current_tool = tk.StringVar()
        self.current_tool.set(BRUSH)

        self.menu = self.create_menu()
        self.tool_panel = self.create_tool_panel(self.current_tool)
        self.canvas = self.create_canvas()
        self.canvas_state = CanvasState(self.canvas, self.model)

        self.styles_box = self.create_styles_box()

        self.root.config(menu=self.menu)
        self.tool_panel.grid(row=0, column=0)
        self.canvas.grid(row=0, column=1)
        self.styles_box.grid(row=0, column=2)

        self.root.mainloop()

    def create_tool_panel(self, tool_var):
        tool_panel = tk.Frame(self.root, height=30)

        for i, tool in enumerate(TOOLS):
            tk.Radiobutton(tool_panel, text=tool, variable=tool_var, value=tool).grid(sticky='W', row=i, column=0)
        return tool_panel

    def create_canvas(self):
        canvas = tk.Canvas(self.root, width=IMG_SIZE[1], height=IMG_SIZE[0], bg='grey')
        canvas.bind("<B1-Motion>", self.paint)
        canvas.bind("<Button-1>", self.click)

        return canvas

    def create_menu(self):
        menu = tk.Menu(self.root)
        menu.add_command(label="Open", command=self.open_file)
        menu.add_command(label="Save...", command=self.save_file)

        return menu

    def open_file(self):
        img_path = fdialog.askopenfilename(title="Select image",
                                           filetypes=(("Images", ["*.jpg", "*.png"]), ("all files", "*.*")))
        img = imread(img_path)
        img = (resize(img, IMG_SIZE) * 256).astype(np.uint8)

        self.canvas_state.update_image(img)

    def save_file(self):
        img_path = fdialog.asksaveasfilename(title="Save file")
        imsave

    def create_styles_box(self):
        box = tk.Frame(self.root)

        scrollbar = tk.Scrollbar(box)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(box)
        listbox.pack()

        for i in range(100):
            listbox.insert(tk.END, i)

        # attach listbox to scrollbar
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)

        return box

    def paint(self, event):
        pass

    def click(self, event):
        x, y = event.x, event.y

        if self.tool == MAGIC:
            self.magic_selection(x, y)

    @property
    def tool(self):
        return self.current_tool.get()

    def magic_selection(self, x, y):
        signature = self.canvas_state.signature
        s_x, s_y = self.canvas_state.to_signature_coords(x, y)

        signature_shape = signature.shape

        selected_signature = signature[s_y, s_x]

        mask = np.zeros([signature_shape[0] - 1, signature_shape[1] - 1])
        for i in range(signature_shape[0] - 1):
            for j in range(signature_shape[1] - 1):
                mask[i, j] = style_angle(signature[i, j], selected_signature)

        threshold = np.where(mask < 0.18)

        selected_signature = signature[threshold[0], threshold[1]].mean(axis=0)

        mask = np.zeros([signature_shape[0] - 1, signature_shape[1] - 1])
        for i in range(signature_shape[0] - 1):
            for j in range(signature_shape[1] - 1):
                mask[i, j] = style_angle(signature[i, j], selected_signature)

        mask = np.where(mask > 0.18, 0.0, 1.0)

        labeled = label(mask)

        mask = np.where(labeled != labeled[s_y, s_x], 0.0, 1.0)

        self.canvas_state.update_mask(resize(mask, self.canvas_state.mask.shape))

        # noinspection PyAttributeOutsideInit


class CanvasState:
    def __init__(self, canvas, model):
        self.model = model
        self.canvas = canvas

    def update_image(self, img):
        self.img = img[:, :, :3]
        self.mask = np.zeros(self.img.shape[:2])

        self.signature = self.model.create_style_signature(self.img, side=25, stride=5)
        self.similarity_matrix = resize(self.model.create_similarity_matrix(signature=self.signature),
                                        self.img.shape[:2])

        self.render_image()

    def render_image(self):
        selector = np.where(self.mask > 0.0)

        masked = self.img.copy()
        masked[selector] = masked[selector] * 0.8 + [256.0 * 0.2, 0.0, 0.0]

        self.pil_img = Image.fromarray(masked.astype('uint8'), 'RGB')
        self.photo = ImageTk.PhotoImage(image=self.pil_img)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def to_signature_coords(self, x, y):
        sx = x / self.img.shape[1] * self.signature.shape[1]
        sy = y / self.img.shape[0] * self.signature.shape[0]
        sx = np.clip(sx, 0, self.signature.shape[1] - 1)
        sy = np.clip(sy, 0, self.signature.shape[0] - 1)
        return int(sx), int(sy)

    def update_mask(self, mask):
        self.mask = mask
        self.render_image()


app = App()
