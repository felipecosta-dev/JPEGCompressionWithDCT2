import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
import time 

import numpy as np

from image_compressor import ImageCompressor

images_save_dir = r'saved_images'

def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
    if file_path:
        block_size = simpledialog.askinteger("Input", "Inserisci l’ampiezza delle finestrelle (F):")
        freq_cutoff = simpledialog.askinteger("Input", "Inserisci la soglia di taglio delle frequenze (d) compreso tra 0 e (2F − 2):")
        process_image(file_path, block_size, freq_cutoff)

def process_image(file_path, F, d):
    global compressor
    compressor = ImageCompressor(file_path, F, d)

    start_time = time.perf_counter()
    compressor.compress()
    end_time = time.perf_counter()

    compression_time = end_time - start_time
    print(f"Compression time: {compression_time:.10f}")

    display_images(file_path, compressor.compressed_image)

    original_label_text.config(text=f"Immagine originale: {compressor.original_filename}")
    processed_label_text.config(text=f"Immagine compressa (F={F}, d={d}): {compressor.compressed_image_filename}")

    original_label_text.grid()
    processed_label_text.grid()
    save_button.grid()

def display_images(original_path, processed):
    canvas = tk.Canvas(frame, width=frame.winfo_screenwidth(), height=frame.winfo_screenheight())
    scroll_y = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scroll_x = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    canvas.grid(row=2, column=0, columnspan=2)
    scroll_y.grid(row=2, column=2, sticky='ns')
    scroll_x.grid(row=3, column=0, columnspan=2, sticky='ew')

    original = Image.open(original_path)
    processed = Image.fromarray(processed.astype(np.uint8))

    max_width = (frame.winfo_screenwidth() // 2) - 20
    ratio = max_width / original.width
    new_height = int(original.height * ratio)

    original = original.resize((max_width, new_height), Image.Resampling.LANCZOS)
    processed_to_display = processed.resize((max_width, new_height), Image.Resampling.LANCZOS)

    original_image_tk = ImageTk.PhotoImage(original)
    processed_image_tk = ImageTk.PhotoImage(processed_to_display)

    canvas.create_image(0, 0, image=original_image_tk, anchor='nw')
    canvas.create_image(original.width + 10, 0, image=processed_image_tk, anchor='nw')

    canvas.config(scrollregion=canvas.bbox("all"))
    canvas.original_image_tk = original_image_tk
    canvas.processed_image_tk = processed_image_tk

def save_compressed_image():
    global compressor
    if compressor and compressor.compressed_image is not None:
        try:
            save_path = compressor.save_compressed_image(images_save_dir)
            messagebox.showinfo("Salvataggio immagine", f"Immagine compressa salvata su {save_path}")
        except Exception as e:
            messagebox.showerror("Salvataggio immagine", f"Non è stato posibile salvare l'immagine. Errore: {e}")

root = tk.Tk()
root.title("DCT Image Compression")
root.state('zoomed')

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

choose_button = ttk.Button(frame, text="Scegli immagine", command=choose_file)
choose_button.grid(row=0, column=0, pady=10)

save_button = ttk.Button(frame, text="Salva immagine compressa", command=save_compressed_image)
save_button.grid(row=0, column=1, pady=10)
save_button.grid_remove() 

original_label_text = ttk.Label(frame, text="", anchor="center")
original_label_text.grid(row=1, column=0, pady=5)
original_label_text.grid_remove() 

processed_label_text = ttk.Label(frame, text="", anchor="center")
processed_label_text.grid(row=1, column=1, pady=5)
processed_label_text.grid_remove()  
