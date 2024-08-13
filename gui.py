import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from manualdct2 import ManualDCT2
from scipydct2 import SciPyDCT2
from utils import apply_frequency_cutoff

def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
    if file_path:
        block_size = simpledialog.askinteger("Input", "Enter block size (F):")
        freq_cutoff = simpledialog.askinteger("Input", "Enter frequency cutoff (d):")
        process_image(file_path, block_size, freq_cutoff)

def process_image(file_path, F, d):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    compressed_image = np.zeros_like(image)

    manual_dct = ManualDCT2(block_size=F)

    for i in range(0, height, F):
        for j in range(0, width, F):
            block = image[i:i+F, j+j+F]
            if block.shape[0] == F and block.shape[1] == F:
                dct_block = manual_dct.dct2(block)
                dct_block = apply_frequency_cutoff(dct_block, d)
                idct_block = manual_dct.idct2(dct_block)
                compressed_image[i:i+F, j:j+F] = np.clip(idct_block, 0, 255)

    display_images(image, compressed_image)

def display_images(original, processed):
    original = Image.fromarray(original)
    processed = Image.fromarray(processed.astype(np.uint8))
    original = ImageTk.PhotoImage(original)
    processed = ImageTk.PhotoImage(processed)

    original_label.config(image=original)
    original_label.image = original
    processed_label.config(image=processed)
    processed_label.image = processed

root = tk.Tk()
root.title("DCT Image Compression")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

choose_button = ttk.Button(frame, text="Choose Image", command=choose_file)
choose_button.grid(row=0, column=0, pady=10)

original_label = ttk.Label(frame)
original_label.grid(row=1, column=0, pady=10)

processed_label = ttk.Label(frame)
processed_label.grid(row=1, column=1, pady=10)

root.mainloop()
