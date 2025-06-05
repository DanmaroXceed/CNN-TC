import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # type: ignore
import cv2  # type: ignore
from keras.models import load_model  # type: ignore
import numpy as np  # type: ignore

modelo = load_model('modelo_tumores.keras')

def cargar_imagen():
    ruta_imagen = filedialog.askopenfilename(title="Seleccionar Imagen", filetypes=[("Archivos de Imagen", "*.png;*.jpg;*.jpeg")])
    if ruta_imagen:
        imagen_original = Image.open(ruta_imagen)
        imagen_original.thumbnail((300, 300))  # Redimensionar la imagen para mostrarla en la interfaz
        imagen_tk = ImageTk.PhotoImage(imagen_original)
        etiqueta_imagen.config(image=imagen_tk)
        etiqueta_imagen.image = imagen_tk  # Evitar que la imagen sea eliminada por el recolector de basura
        I = cv2.imread(ruta_imagen)
        I_resz = cv2.resize(I,dsize=(150,150), interpolation=cv2.INTER_CUBIC)
        I_resz = I_resz.astype('float')

        if round(modelo.predict(np.array([I_resz]))[0][0])==1:
            texto_sobre_imagen = "Positivo"
            print("Positivo")
            etiqueta_texto.config(text=texto_sobre_imagen)
        else:
            texto_sobre_imagen = "Negativo"
            print("Negativo")
            etiqueta_texto.config(text=texto_sobre_imagen)


# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Procesador de Imágenes")

# Botón para cargar la imagen
boton_cargar = tk.Button(ventana, text="Cargar Imagen", command=cargar_imagen)
boton_cargar.pack(pady=10)

# Etiqueta para mostrar el texto sobre la imagen
etiqueta_texto = tk.Label(ventana, text="")
etiqueta_texto.pack(pady=10)

# Etiqueta para mostrar la imagen
etiqueta_imagen = tk.Label(ventana)
etiqueta_imagen.pack()

# Iniciar el bucle principal
ventana.mainloop()

