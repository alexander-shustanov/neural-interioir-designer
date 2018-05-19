import glob
import tkinter as tk
import tkinter.filedialog as fdialog
from os.path import basename

import numpy as np
from PIL import Image, ImageTk
from skimage.io import imread, imsave
from skimage.measure import label
from skimage.transform import resize

from nn import Model, style_angle
from paint_tool import PaintTool
from style_helper import StyleHelper

BRUSH = "brush"
MAGIC = "magic"
STYLE = "style"

IMG_SIZE = (250, 250)

THR = 0.2

TOOLS = [BRUSH, MAGIC, STYLE]


class App:
    def __init__(self):
        self.model = Model()

        self.root = tk.Tk()
        self.root.config()

        self.current_tool = tk.StringVar()
        self.current_tool.set(BRUSH)

        self.menu = self.create_menu()
        self.tool_panel = self.create_tool_panel(self.current_tool)
        self.canvas = self.create_canvas()
        self.canvas_state = CanvasState(self.canvas, self.model, self.on_mask_changed)

        self.styles_box = self.create_styles_box()

        self.root.grid_rowconfigure(0, weight=0)

        self.root.config(menu=self.menu)
        self.tool_panel.grid(row=0, column=0)
        self.canvas.grid(row=0, column=1)
        self.styles_box.grid(row=0, column=2)

        self.tool_changed()

        self.keys = KeysHelper(self.root)

        self.paint_tool = PaintTool(self.canvas_state, self.keys, IMG_SIZE)

        self.root.mainloop()

    def on_mask_changed(self):
        self.style_helper = None

    def create_tool_panel(self, tool_var):
        tool_panel = tk.Frame(self.root, height=30)

        for i, tool in enumerate(TOOLS):
            tk.Radiobutton(tool_panel, text=tool, variable=tool_var, value=tool, command=self.tool_changed) \
                .grid(sticky='W', row=i, column=0)
        return tool_panel

    def create_canvas(self):
        canvas = tk.Canvas(self.root, width=IMG_SIZE[1], height=IMG_SIZE[0], bg='grey')
        canvas.bind("<B1-Motion>", self.paint)
        canvas.bind("<Button-1>", self.click)
        canvas.bind("<ButtonRelease-1>", self.mouse_release)

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
        imsave(img_path, self.canvas_state.img)

    def create_styles_box(self):
        styles = sorted(glob.glob("styles/*.*"))

        box = tk.Frame(self.root)

        scrollbar = tk.Scrollbar(box, orient=tk.VERTICAL)

        listbox = tk.Listbox(box, selectmode=tk.SINGLE)
        listbox.pack(side=tk.LEFT)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        for style in styles:
            listbox.insert(tk.END, basename(style))

        # attach listbox to scrollbar
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)

        style_image_label = tk.Label(box)
        style_image_label.pack()

        apply_button = tk.Button(box, text="Apply style", command=self.apply_style)
        apply_button.pack()

        def on_select(event):
            self.style_helper = None

            selected = int(listbox.curselection()[0])

            self.style_image_pil = Image.open(styles[selected]).resize(IMG_SIZE)
            self.style_photo = ImageTk.PhotoImage(image=self.style_image_pil)

            style_image_label.configure(image=self.style_photo)
            style_image_label.image = self.style_photo

            self.style_image_name = styles[selected]

        listbox.bind('<<ListboxSelect>>', on_select)

        return box

    def paint(self, event):
        if self.tool == BRUSH:
            self.paint_tool.paint(event.x, event.y)

    def mouse_release(self, *args):
        if self.tool == BRUSH:
            self.paint_tool.end()

    def click(self, event):
        x, y = event.x, event.y

        if self.tool == MAGIC:
            self.magic_selection(x, y)

    def tool_changed(self):
        self.styles_box.grid_remove()

        if self.tool == STYLE:
            self.styles_box.grid()

    def apply_style(self):
        if self.style_helper is None:
            style_image = resize(imread(self.style_image_name), IMG_SIZE) * 256.0
            self.style_helper = StyleHelper(self.model, self.canvas_state.img, style_image,
                                            np.where(self.canvas_state.mask > 0.0, 1.0, 0.0))

        mixed = self.canvas_state.img.copy().astype(float)
        for i in range(10):
            mixed -= self.style_helper.step(mixed)
            mixed = np.clip(mixed, 0.0, 255.0)

        self.canvas_state.img = mixed.astype(np.uint8)

        self.canvas_state.render_image()

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

        threshold = np.where(mask < THR)

        if len(threshold[0]) == 0:
            return

        selected_signature = signature[threshold[0], threshold[1]].mean(axis=0)

        mask = np.zeros([signature_shape[0] - 1, signature_shape[1] - 1])
        for i in range(signature_shape[0] - 1):
            for j in range(signature_shape[1] - 1):
                mask[i, j] = style_angle(signature[i, j], selected_signature)

        mask = np.where(mask > THR, 0.0, 1.0)

        labeled = label(mask)

        mask = np.where(labeled != labeled[min(s_y, mask.shape[0] - 1), min(s_x, mask.shape[1] - 1)], 0.0, 1.0)

        mask = resize(mask, self.canvas_state.mask.shape)

        if self.keys.ctrl:
            self.canvas_state.add_mask(mask)
        elif self.keys.alt:
            self.canvas_state.remove_mask(mask)
        else:
            self.canvas_state.update_mask(mask)

        # noinspection PyAttributeOutsideInit


class CanvasState:
    def __init__(self, canvas, model, mask_changed_handler):
        self.mask_changed_handler = mask_changed_handler
        self.model = model
        self.canvas = canvas

    def update_image(self, img):
        self.img = img[:, :, :3]
        self.mask = np.zeros(self.img.shape[:2])

        self.signature = self.model.create_style_signature(self.img, side=25, stride=5)
        self.similarity_matrix = resize(self.model.create_similarity_matrix(signature=self.signature),
                                        self.img.shape[:2])

        self.render_image()

        self.mask_changed_handler()

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

    def add_mask(self, mask):
        self.mask[np.where(mask > 0.0)] = 1.0
        self.render_image()

    def remove_mask(self, mask):
        self.mask[np.where(mask > 0.0)] = 0.0
        self.render_image()

    def update_mask(self, mask):
        self.mask = mask
        self.render_image()

    @property
    def is_mask_empty(self):
        return len(np.where(self.mask > 0.0)[0]) == 0


class KeysHelper:
    def __init__(self, root):
        self._alt = False
        self._ctrl = False

        root.bind("<Alt_L>", self.on_alt)
        root.bind("<Control_L>", self.on_ctrl)
        root.bind("<KeyRelease-Alt_L>", self.out_alt)
        root.bind("<KeyRelease-Control_L>", self.out_ctrl)

    def on_alt(self, event):
        self._alt = True
        print("Press Alt")

    def on_ctrl(self, event):
        self._ctrl = True
        print("Press Ctrl")

    def out_alt(self, event):
        self._alt = False
        print("Release Alt")

    def out_ctrl(self, event):
        self._ctrl = False
        print("Release Ctrl")

    @property
    def ctrl(self):
        return self._ctrl

    @property
    def alt(self):
        return self._alt


app = App()
