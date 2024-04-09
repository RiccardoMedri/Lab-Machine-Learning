import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid  


class Image_Manager:

    @staticmethod
    def read_as_color(path, type: int):
        result = None
        if type == 1:
            read_type = cv2.IMREAD_COLOR
            result = Image_Manager.read_generic(path, read_type)
        elif type == 2:
            read_type = cv2.IMREAD_GRAYSCALE
            result = Image_Manager.read_generic(path, read_type)
        elif type == 3:
            read_type = cv2.IMREAD_UNCHANGED
            result = Image_Manager.read_generic(path, read_type)
        return result

    @staticmethod
    def read_generic(*args):
        if isinstance(args[0], str):
            image = cv2.imread(args[0])
            if image is not None:
                return image
            else:
                print("Immagine non leggibile. Controllare il percorso indicato")
                return None
        elif isinstance(args[0], list):
            images_list = []
            for percorso in args[0]:
                image = cv2.imread(percorso)
                if image is not None:
                    images_list.append(image)
                else:
                    print(f"Immagine con percorso {percorso} non leggibile. Controllare il percorso indicato")
                    return None
            return images_list
        else:
            print("Valore non valido")
            return None
        
    @staticmethod
    def to_disk(image : np.ndarray, name : str, extension : str, path : str):
        complete_path = path + name + extension
        saving = cv2.imwrite(complete_path, image)
        if (saving):
            print("Salvataggio avvenuto correttamente")
            return complete_path
        else:
            print("Errore in fase di salvataggio, controllare i valori inseriti")

    @staticmethod
    def get_info(image):
        if isinstance(image, np.ndarray):
            Image_Manager.__print(image)
        elif isinstance(image, list):
            for immagine in image:
                Image_Manager.__print(immagine)
        else:
            print("Valore non valido")
            return None

    @staticmethod
    def show(image):
        if isinstance(image, np.ndarray):   
            cv2.imshow("window", image)
        elif isinstance(image, list):
            fig = plt.figure(figsize=(15,15))
            colonne = 4
            righe = math.ceil(len(image) / colonne)
            grid = ImageGrid(fig, 111, nrows_ncols=(righe, colonne), axes_pad=0.1)
            for ax, im in zip(grid, image):
                ax.imshow(im)
            plt.show()
        else:
            print("Valore non valido")
            return None
                
    @staticmethod
    def as_rgb(image):
        if isinstance(image, np.ndarray):
            image_as_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_as_rgb
        elif isinstance(image, list):
            image_list = []
            for immagine in image:
                image_as_rgb = cv2.cvtColor(immagine, cv2.COLOR_BGR2RGB)
                image_list.append(image_as_rgb)
        else:
            print("Valore non valido")
            return None

    @staticmethod
    def as_bgr(image):
        if isinstance(image, np.ndarray):
            image_as_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image_as_bgr
        elif isinstance(image, list):
            image_list = []
            for immagine in image:
                image_as_bgr = cv2.cvtColor(immagine, cv2.COLOR_RGB2BGR)
                image_list.append(image_as_bgr)
        else:
            print("Valore non valido")
            return None
        
    @staticmethod
    def get_channels(*args):
        if isinstance(args[0], np.ndarray):
            blue, green, red = cv2.split(args[0])
            if len(args) == 1:  
                return blue, green, red
            elif args[1] == "blue":
                return blue
            elif args[1] == "green":
                return green
            elif args[1] == "red":
                return red
        elif isinstance(args[0], list):
            images_list = []
            for immagine in args[0]:
                blue, green, red = cv2.split(immagine)
                images_list.append((blue, green, red))
            return images_list
        else:
            print("Valore non valido")
            return None

    @staticmethod
    def merge_channel(blue, green, red):
        image = cv2.merge([blue, green, red])
        return image

    @staticmethod
    def __print(image: np.ndarray):
        print(f'Dimensioni della matrice:\t{image.shape}')
        print(f'Larghezza:\t\t\t{image.shape[0]}')
        print(f'Altezza:\t\t\t{image.shape[1]}')
        print(f'Numero di canali:\t\t{image.shape[2]}')






def main():
    image_path = './imgs/dark.png'
    image_list = ['./imgs/1.JPG', './imgs/2.jpg', './imgs/3.jpg', './imgs/4.jpg', './imgs/5.jpg', './imgs/dark.png', './imgs/kitten.png', './imgs/light.png']

    immagine1 = Image_Manager.read_as_color(image_path, 3)
    cv2.imshow("window", immagine1)
    cv2.waitKey(0)
    lista_immagini1 = Image_Manager.read_as_color(image_list, 1)

    percorso_immagine = Image_Manager.to_disk(immagine1, "PROVA", ".png", "./imgs/prove/")
    immagine4 = Image_Manager.read_as_color(percorso_immagine, 1)
    cv2.imshow("window", immagine4)
    cv2.waitKey(0)

    immagine5 = Image_Manager.read_as_color(image_path, 1)
    Image_Manager.get_info(immagine5)

    Image_Manager.show(lista_immagini1)

    immagine6 = Image_Manager.as_bgr(immagine1)
    cv2.imshow("window", immagine6)
    cv2.waitKey(0)

    immagine7 = Image_Manager.as_rgb(immagine1)
    cv2.imshow("window", immagine7)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()