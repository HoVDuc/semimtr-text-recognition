import os
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

'''
- Open data: Mở file train_gt
- Save file: Mở file txt để lưu
- Điền tên file vào image name để load ảnh
- Nhấn Right để Next
- Nhấn Left để Prev
- Nhấn Up để Save
'''

class GUI:

    def __init__(self, path) -> None:
        self.idx = -1
        self.path = path
        self.root = tk.Tk()
        self.root.title("Labeling")
        self.list_canvas = []
        self.root.geometry('600x500')
        self.create_canvas()
        self.create_button()
        self.binding()
        self.create_entry()
        self.create_label()
        self.create_text()
        self.set_position()

    def clear(self):
        self.entry_gt.delete(0, 'end')
        self.entry_idx.delete(0, 'end')
        self.entry_image_path.delete(0, 'end')

    def setidx(self, i):
        self.clear()
        self.idx += i
        self.update()

    def create_entry(self):
        self.entry_data_path = ttk.Entry(self.canvas_data)
        self.entry_save_path = ttk.Entry(self.canvas_save)
        self.entry_image_path = ttk.Entry(self.canvas_path)
        self.entry_gt = ttk.Entry(self.canvas_gt)
        self.entry_idx = ttk.Entry(self.canvas_gt, width=5)

    def create_button(self):
        self.btn_data = ttk.Button(
            self.canvas_data, text='Open data', command=lambda: self.open_file(self.entry_data_path))
        self.btn_file_save = ttk.Button(
            self.canvas_save, text='Save file', command=lambda: self.open_file(self.entry_save_path))
        # TODO: set load function
        self.btn_load = ttk.Button(
            self.canvas_path, text='Load', command=self.load)
        self.btn_prev = ttk.Button(
            self.canvas_button, text='Prev', command=lambda: self.setidx(-1))
        self.btn_next = ttk.Button(
            self.canvas_button, text='Next', command=lambda: self.setidx(1))
        self.btn_save = ttk.Button(
            self.canvas_button, text='Save', command=self.save)

    def binding(self):
        self.root.bind('<Left>', lambda event: self.setidx(-1))
        self.root.bind('<Right>', lambda event: self.setidx(1))
        self.root.bind('<Up>', lambda event: self.save())

    def create_label(self):
        self.image_path = tk.Label(self.canvas_path, text='image name')
        self.gt_label = tk.Label(self.canvas_gt, text='gt')

    def create_text(self):
        self.text = tk.Text(self.canvas_text, width=45, height=15)

    def set_position(self):
        # position open file
        self.canvas_data.pack()
        self.btn_data.pack(side=tk.LEFT, padx=(10, 10))
        self.entry_data_path.pack()

        # position open save
        self.canvas_save.pack(pady=(10, 10))
        self.btn_file_save.pack(side=tk.LEFT, padx=(10, 10))
        self.entry_save_path.pack()

        # position display image
        self.display_canvas.pack(pady=(10, 10))
        self.canvas_path.pack(pady=(10, 10))
        self.image_path.pack(side=tk.LEFT, padx=(0, 10))
        self.entry_image_path.pack(side=tk.LEFT, padx=(10, 10))
        self.btn_load.pack(side=tk.LEFT, padx=(10, 0))
        self.canvas.pack()
        self.canvas_gt.pack()
        self.gt_label.pack(side=tk.LEFT, padx=(10, 10))
        self.entry_gt.pack(side=tk.LEFT, padx=(10, 10))
        self.entry_idx.pack(side=tk.LEFT, padx=(10, 10))

        # postion button
        self.canvas_button.pack(pady=(10, 10))
        self.btn_prev.pack(side=tk.LEFT, padx=(0, 10))
        self.btn_save.pack(side=tk.LEFT, padx=(0, 0))
        self.btn_next.pack(side=tk.LEFT, padx=(10, 0))

        # position display text
        self.canvas_text.pack(pady=(10, 10))
        self.text.pack()

    def open_file(self, entry):
        f = filedialog.askopenfilename()
        try:
            entry.delete(0, 'end')
        except:
            pass
        entry.insert(INSERT, f)
        if entry == self.entry_data_path:
            with open(f, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            lines = [line.split('\t') for line in lines]
            df = pd.DataFrame(lines, columns=['image_path', 'gt'])
            df['gt'] = df['gt'].apply(lambda x: x.replace('\n', ''))
            self.data = df
        else:
            self.name_file = f
            with open(self.name_file, 'r+', encoding='utf-8') as file:
                self.text.insert(INSERT, file.read())

    def create_canvas(self):
        self.canvas_data = tk.Canvas(self.root)
        self.canvas_save = tk.Canvas(self.root)
        self.display_canvas = tk.Canvas(self.root)
        self.canvas = tk.Canvas(self.display_canvas,
                                bg='black', height=32, width=100)
        self.canvas_path = tk.Canvas(self.display_canvas)
        self.canvas_gt = tk.Canvas(self.display_canvas)
        self.canvas_button = tk.Canvas(self.root)
        self.canvas_text = tk.Canvas(self.root)

    def load(self):
        if self.entry_image_path.get():
            self.idx = self.data[self.data['image_path']
                                 == self.entry_image_path.get()].index[0]
        else:
            tk.messagebox.showerror(title='Lỗi!', message='Lỗi!')
        self.clear()
        self.update()

    def save(self):
        image_path = self.data['image_path'].iloc[self.idx]
        with open(self.name_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line.split('\t') for line in lines]
        df = pd.DataFrame(lines, columns=['image_path', 'gt'])
        if df['image_path'].isin([image_path]).any():
            tk.messagebox.showerror(
                title='Lỗi!', message='Đường dẫn đã tồn tại')
        else:
            with open(self.name_file, 'a+', encoding='utf-8') as f:
                f.write('{}\t{}\n'.format(
                    self.data['image_path'].iloc[self.idx], self.entry_gt.get()))
            with open(self.name_file, 'r', encoding='utf-8') as f:
                self.text.delete('1.0', 'end')
                self.text.insert(INSERT, f.read())

    def update(self):
        try:
            self.list_canvas.pop().destroy()
        except:
            pass
        data = self.data.iloc[self.idx]
        self.entry_idx.insert(INSERT, self.idx)
        path = os.path.join(self.path, data['image_path'])
        image = Image.open(path)
        w, h = image.size
        new_canvas = tk.Canvas(self.canvas, width=w, height=h, bg='red')
        self.photo = ImageTk.PhotoImage(image)
        self.entry_image_path.insert(0, data['image_path'])
        self.entry_gt.insert(0, data['gt'])
        new_canvas.create_image(w//2 + 2, 2, anchor='n', image=self.photo)
        self.list_canvas.append(new_canvas)
        new_canvas.pack()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = '../Datasets/new_train/'
    gui = GUI(root)
    gui.run()
