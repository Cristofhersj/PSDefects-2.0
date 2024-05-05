import tkinter as tk
import os
import subprocess

def close_program():
    root.destroy()

def open_directory():
    directory = "/home/vioro/Desktop/Resultados"  # replace with the path to your directory
    subprocess.call(["xdg-open", directory])

root = tk.Tk()
root.configure(bg='#340623')

# make the window fullscreen
root.attributes("-fullscreen", True)

# bind the <Escape> event to the root window
root.bind("<Escape>", lambda event: close_program())

# create a frame for the status label
status_frame = tk.Frame(root, bg='#340623')
status_frame.pack(fill=tk.BOTH, padx=30, pady=10)

# create a label for the status text
status_label = tk.Label(status_frame, text="Estado Actual: Inicio",
                         font=('Helvetica', 40), fg='#F6F6F5', bg='#340623')
status_label.pack(pady=20, anchor=tk.CENTER)

# create a frame for the buttons
button_frame = tk.Frame(root, bg='#340623')
button_frame.pack(fill=tk.BOTH, expand=True, padx=80)

# create the buttons
ayuda_button = tk.Button(button_frame, text="?", font=('Helvetica', 40),
                          bg='#340623', fg='#F6F6F5', relief='flat', bd=0,
                          command=lambda: print("Ayuda"))
ayuda_button.pack(side=tk.RIGHT, padx=10, pady=10)

imagenes_button = tk.Button(button_frame, text="[+] Borrar imágenes", font=('Helvetica', 40),
                            bg='#340623', fg='#F6F6F5', relief='flat', bd=0,
                            command=open_directory)
imagenes_button.pack(side=tk.RIGHT, padx=10, pady=10)

iniciar_button = tk.Button(button_frame, text="[Enter] Iniciar Revisión", font=('Helvetica', 40),
                           bg='#340623', fg='#F6F6F5', relief='flat', bd=0,
                           command=lambda: print("Iniciar Revisión"))
iniciar_button.pack(side=tk.RIGHT, padx=10, pady=10)

cerrar_button = tk.Button(button_frame, text="[Esc] Cerrar", font=('Helvetica', 40),
                          bg='#340623', fg='#F6F6F5', relief='flat', bd=0,
                          command=close_program)
cerrar_button.pack(side=tk.RIGHT, padx=10, pady=10)

root.mainloop()