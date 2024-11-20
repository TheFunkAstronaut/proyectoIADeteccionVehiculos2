# resize_images.py en la carpeta utilidades/
from PIL import Image
import os
import random
import shutil

def move_images_to_test(input_folder, test_folder, test_ratio=0.15):
    """
    Mueve un porcentaje de imágenes de la carpeta `input_folder` a `test_folder`.
    """
    os.makedirs(test_folder, exist_ok=True)
    all_images = os.listdir(input_folder)
    test_size = int(len(all_images) * test_ratio)
    test_images = random.sample(all_images, test_size)

    for filename in test_images:
        shutil.move(os.path.join(input_folder, filename), os.path.join(test_folder, filename))


def resize_images_in_folder(input_folder, output_folder, size=(224, 224)):
    """
    Redimensiona todas las imágenes de la carpeta `input_folder` y las guarda en `output_folder`.
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            img = img.resize(size)
            img.save(os.path.join(output_folder, filename))


if __name__ == "__main__":
    # Directorios originales
    base_dir = '/home/not_funker/PycharmProjects/proyectoIADeteccionVehiculos/data'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Subcarpetas por clase
    classes = ['auto', 'bicicleta', 'camion', 'motocicleta']

    for class_name in classes:
        # Directorios de entrada y salida
        class_train_dir = os.path.join(train_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)

        # Mover el 15% de imágenes a `test`
        move_images_to_test(class_train_dir, class_test_dir)

        # Redimensionar imágenes en `train` y `test`
        resize_images_in_folder(class_train_dir, f"{class_train_dir}_resized")
        resize_images_in_folder(class_test_dir, f"{class_test_dir}_resized")

    #resize_images_in_folder('data/train/bicicleta', 'data_resized/train/bicicleta')
    #resize_images_in_folder('data/train/camion', 'data_resized/train/camion')
    #resize_images_in_folder('data/train/motocicleta', 'data_resized/train/motocicleta')

    #resize_images_in_folder('data/test/auto', 'data_resized/test/auto')
    #resize_images_in_folder('data/test/bicicleta', 'data_resized/test/bicicleta')
    #resize_images_in_folder('data/test/camion', 'data_resized/test/camion')
    #resize_images_in_folder('data/test/motocicleta', 'data_resized/test/motocicleta')


