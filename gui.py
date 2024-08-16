import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from manualdct2 import ManualDCT2
from scipydct2 import SciPyDCT2
from utils import apply_frequency_cutoff
import os 

def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
    if file_path:
        block_size = simpledialog.askinteger("Input", "Inserisci l’ampiezza delle finestrelle (F):")
        freq_cutoff = simpledialog.askinteger("Input", "Inserisci la soglia di taglio delle frequenze (d) compreso tra 0 e (2F − 2):")
        process_image(file_path, block_size, freq_cutoff)

def process_image(file_path, F, d):
    global compressed_image, original_filename, original_directory, block_size, freq_cutoff
    original_filename = os.path.splitext(os.path.basename(file_path))[0]
    original_directory = os.path.dirname(file_path)
    block_size = F
    freq_cutoff = d

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    compressed_image = np.zeros_like(image)

    dct_engine = SciPyDCT2()

    for i in range(0, height, F):
        for j in range(0, width, F):
            block = image[i:i+F, j:j+F]
            if block.shape[0] == F and block.shape[1] == F:
                dct_block = dct_engine.dct2(block)
                dct_block = apply_frequency_cutoff(dct_block, d)
                idct_block = dct_engine.idct2(dct_block)
                compressed_image[i:i+F, j:j+F] = np.clip(idct_block, 0, 255)

    display_images(image, compressed_image, F, d)

def display_images(original, processed, F, d):
    original = Image.fromarray(original)
    processed = Image.fromarray(processed.astype(np.uint8))
    original = ImageTk.PhotoImage(original)
    processed = ImageTk.PhotoImage(processed)

    original_label.config(image=original)
    original_label.image = original
    original_label_text.config(text="Originale")

    processed_label.config(image=processed)
    processed_label.image = processed
    processed_label_text.config(text=f"Compressa (F={F}, d={d})")

    original_label_text.grid()
    processed_label_text.grid()
    save_button.grid()

def save_compressed_image():
    if compressed_image is not None:
        filename = f"{original_filename}_f{block_size}_d{freq_cutoff}.bmp"
        cv2.imwrite(filename, compressed_image)
        tk.messagebox.showinfo("Immagine salvata", f"Immagine compressa salvata come {filename}")

root = tk.Tk()
root.title("DCT Image Compression")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

choose_button = ttk.Button(frame, text="Scegli immagine", command=choose_file)
choose_button.grid(row=0, column=0, pady=10)

original_label_text = ttk.Label(frame, text="", anchor="center")
original_label_text.grid(row=1, column=0, pady=5)
original_label_text.grid_remove() 

original_label = ttk.Label(frame)
original_label.grid(row=2, column=0, padx=10, pady=10)

processed_label_text = ttk.Label(frame, text="", anchor="center")
processed_label_text.grid(row=1, column=1, pady=5)
processed_label_text.grid_remove()  

processed_label = ttk.Label(frame)
processed_label.grid(row=2, column=1, padx=10, pady=10)

save_button = ttk.Button(frame, text="Salva immagine compressa", command=save_compressed_image)
save_button.grid(row=3, column=1, pady=10)
save_button.grid_remove() 
