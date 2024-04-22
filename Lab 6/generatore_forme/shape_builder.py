import cv2
import random
import numpy as np

# Queste costanti le uso per dare un colore diverso ad ogni forma
COLORS = [(147, 112, 195), 
          (241, 192, 4), 
          (232, 140, 232), 
          (55, 172, 253), 
          (199, 217, 131), 
          (20, 2, 182), 
          (207, 114, 249), 
          (33, 89, 150), 
          (132, 49, 144), 
          (44, 153, 208)]

# Queste costanti le uso per dare un nome diverso ad ogni forma.
NAMES = ['', 
         'circle', 
         'triangle', 
         "square",
         "pentagon",
         "hexagon",
         "heptagon",
         "octagon",
         "nonagon",
         "decagon"]


class ShapeBuilder:
    """
    Classe dedicata a fornire punti per il disegno di pologoni.
    """

    def __init__(self, cfg : object) -> None:   
        
        # Anche questa classe si configura tramite il json.    
        # Il disegno sara' su una tela nera grande (w*h)-
        # Il centro della tela sara' in (ctr_x, ctr_y).
        self.w = cfg.canvas_width
        self.h = cfg.canvas_height
        self.ctr_x = self.w // 2
        self.ctr_y = self.h // 2
        self.min_sides = cfg.min_sides
        self.max_sides = cfg.max_sides
        self.colored = cfg.colored
        
    def get_class_names(self) -> list[str]:
        
        class_names = []
        for i in range(1 + self.max_sides - self.min_sides):
            class_names.append(NAMES[i + self.min_sides - 1])
            
        return class_names

    def create_synthetic_object(self) -> tuple[np.ndarray, str]:

        # Ogni volta che viene chiesto di creare un oggetto, decido
        # in maniera casuale quanti lati avra'.
        sides = random.randint(self.min_sides, self.max_sides)
        
        # In base ai lati, seleziono colore e nome da assegnargli.
        canvas = np.zeros([self.w, self.h, 3 if self.colored else 1])
        color = COLORS[sides-1] if self.colored else int(sum(COLORS[sides-1])/len(COLORS[sides-1]))

        if sides == 2:
            center, radius = self.random_circle()
            cv2.circle(canvas, center, radius, color, -1)
        else:
            pts = self.random_regular_poly(sides)
            cv2.fillPoly(canvas, np.array([pts]), color)

        return canvas, NAMES[sides-1]

    def random_regular_poly(self, sides : int) -> list[tuple[int, int]] | None:
        """
        Restituisce, se possibile, i vertici del poligono regolare richiesto.

        Args:
            sides (int): Numero di lati del poligono.

        Returns:
            list[tuple[int, int]] | None: Vertici del poligono o None.
        """
        
        # Le richieste valide partono dal triangolo.
        if sides < 3:
            return None
        else:

            # Trovo centro e raggio del cerchio casuale che circoscrivera' il poligono.
            center, radius = self.random_circle()

            # Definisco gli angoli di rotazione che creeranno i vertici.
            angles = np.arange(0, 360, 360 // sides)

            # Definisco il primo vertice e lo ruoto di un angolo casuale.
            start_point = (center[0] + radius, center[1])        
            start_point = self.rotate_around_pivot(start_point, center, random.randint(0, 360))

            # Creo i successivi vertici ruotando dell'angolo necessario.
            pts = []
            for a in angles:
                pts.append(self.rotate_around_pivot(start_point, center, a))
            return pts

    def random_circle(self) -> tuple[tuple[int, int], int]:
        """
        Restituisce un cerchio casuale nel piano di lavoro.

        Returns:
            tuple[tuple[int, int], int]: Centro e raggio del cerchio casuale.
        """

        # Creo un centro casuale su una pozione ridotta dell'area di lavoro.
        center = self.random_center(self.ctr_x - self.w // 4, self.ctr_x + self.w // 4,
                                    self.ctr_y - self.h // 4, self.ctr_y + self.h // 4)
        
        # Creo un raggio casuale.
        max_radius = min(self.w // 4, self.h // 4)
        radius = self.random_radius(int(0.75 * max_radius), max_radius)

        return center, radius

    def random_center(self, min_x : int, max_x : int, min_y : int, max_y : int) -> tuple[int, int]:
        """
        Richiede un punto nell'area di lavoro definita.

        Args:
            min_x (int): Minima x dell'area di lavoro.
            max_x (int): Massima x dell'area di lavoro.
            min_y (int): Minima y dell'area di lavoro.
            max_y (int): Massima y dell'area di lavoro.

        Returns:
            tuple[int, int]: Punto.
        """        
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        return (x, y)
    
    def random_radius(self, min_l : int, max_l : int) -> int:
        """
        Richiede un valore compreso fra i due estremi.

        Args:
            min_l (int): Minimo
            max_l (int): Massimo

        Returns:
            int: Valore causale.
        """        
        return random.randint(min_l, max_l)

    def rotate_around_pivot(self, 
                            pts : list[tuple[int, int]], 
                            pivot : tuple[int, int], 
                            degrees : int, 
                            as_int = True) -> tuple[int, int] | tuple[float, float]:
        """
        Ruota una lista di punti di un angolo attorno ad un punto.

        Args:
            pts (list[tuple[int, int]]): Lista punti da ruotare.
            pivot (tuple[int, int]): Punto di rotazione.
            degrees (int): Angolo di rotazione.
            as_int (bool, optional): Indica se restituire valori interi. Defaults to True.

        Returns:
            tuple[int, int] | tuple[float, float]: _description_
        """        

        # Converte l'angolo da gradi a radianti.
        angle = np.deg2rad(degrees)

        # Crea la matrice di rotazione a partire dell'angolo scelto.
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        
        # Calcola i punti ruotati.
        pvt = np.atleast_2d(pivot)
        pts = np.atleast_2d(pts)
        rotated_pts = np.squeeze((R @ (pts.T - pvt.T) + pvt.T).T)

        # Restituisce i valori.
        return rotated_pts.astype(int) if as_int else rotated_pts

if __name__ == "__main__":
    pass