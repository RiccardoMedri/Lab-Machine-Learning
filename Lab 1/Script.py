import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid  


class Image_Manager:

    @staticmethod
    def read_as_color(path):
        if isinstance(path, str):
            coloured_image = cv2.imread(path, cv2.IMREAD_COLOR)
            if coloured_image is not None:
                return coloured_image
            else:
                print("Immagine non leggibile. Controllare il percorso indicato")
                return None
        elif isinstance(path, list):
            images_list = []
            for percorso in path:
                coloured_image = cv2.imread(percorso, cv2.IMREAD_COLOR)
                if coloured_image is not None:
                    images_list.append(coloured_image)
                else:
                    print(f"Immagine con percorso {percorso} non leggibile. Controllare il percorso indicato")
                    return None
            return images_list
        else:
            print("Valore non valido")
            return None

    @staticmethod
    def read_as_mono(path):
        if isinstance(path, str):
            grascale_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if grascale_image is not None:
                return grascale_image
            else:
                print("Immagine non leggibile. Controllare il percorso indicato")
                return None
        elif isinstance(path, list):
            images_list = []
            for percorso in path:
                grascale_image = cv2.imread(percorso, cv2.IMREAD_GRAYSCALE)
                if grascale_image is not None:
                    images_list.append(grascale_image)
                else:
                    print(f"Immagine con percorso {percorso} non leggibile. Controllare il percorso indicato")
                    return None
            return images_list
        else:
            print("Valore non valido")
            return None

    @staticmethod
    def read_as_js(path):
        if isinstance(path, str):
            unchanged_image = cv2.imread(path)
            if unchanged_image is not None:
                return unchanged_image
            else:
                print("Immagine non leggibile. Controllare il percorso indicato")
                return None
        elif isinstance(path, list):
            images_list = []
            for percorso in path:
                unchanged_image = cv2.imread(percorso)
                if unchanged_image is not None:
                    images_list.append(unchanged_image)
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
            colonne = len(image)
            righe = math.ceil(colonne / 4)
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
    def __print(image: np.ndarray):
        print(f'Dimensioni della matrice:\t{image.shape}')
        print(f'Larghezza:\t\t\t{image.shape[0]}')
        print(f'Altezza:\t\t\t{image.shape[1]}')
        print(f'Numero di canali:\t\t{image.shape[2]}')

def main():
    image_path = './imgs/dark.png'
    image_list = ['./imgs/dark.png', './imgs/kitten.png', './imgs/light.png']

    immagine1 = Image_Manager.read_as_color(image_path)
    cv2.imshow("window", immagine1)
    cv2.waitKey(0)
    lista_immagini1 = Image_Manager.read_as_color(image_list)

    immagine2 = Image_Manager.read_as_mono(image_path)
    cv2.imshow("window", immagine2)
    cv2.waitKey(0)

    immagine3 = Image_Manager.read_as_js(image_path)
    cv2.imshow("window", immagine3)
    cv2.waitKey(0)

    percorso_immagine = Image_Manager.to_disk(immagine1, "PROVA", ".png", "./imgs/prove/")
    immagine4 = Image_Manager.read_as_color(percorso_immagine)
    cv2.imshow("window", immagine4)
    cv2.waitKey(0)

    immagine5 = Image_Manager.read_as_color(image_path)
    Image_Manager.get_info(immagine5)

    Image_Manager.show(lista_immagini1)

    immagine6 = Image_Manager.as_bgr(immagine1)
    cv2.imshow("window", immagine6)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()