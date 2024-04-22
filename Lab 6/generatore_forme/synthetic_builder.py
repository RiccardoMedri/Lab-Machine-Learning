import sys
import cv2
import json
import shutil
import jsonschema # conda install jsonschema
import numpy as np

from pathlib import Path
from types import SimpleNamespace
from shape_builder import ShapeBuilder


class SyntheticBuilder:

    def __init__(self, cfg : object) -> None:       
    
        self.out_path = Path(cfg.io.out_folder)
        
        # Il percorso indicato esiste?
        if self.out_path.exists():
            
            # E' una cartella valida?
            if not self.out_path.is_dir():
                print(f'Directory {self.out_path.as_posix()} is not a valid folder')
                sys.exit(-1)
            
            # Se ha del contenuto, lo rimuovo.
            if not next(self.out_path.iterdir(), None) is None:
                shutil.rmtree(self.out_path)  

        # Ricreo la cartella di output.
        self.out_path.mkdir()
        
        # Creo altre 3 cartelle all'interno della cartella di destinazione.
        # - training
        # - test
        # - validazione
        
        self.tr_path = self.out_path / 'training'
        self.tr_path.mkdir()
        
        self.te_path = self.out_path / 'test'
        self.te_path.mkdir()
        
        self.va_path = self.out_path / 'validation'
        self.va_path.mkdir()

        # La percentuale di immagini di test e di validazione non devono sforare il 100% del totale.        
        if (cfg.synth_generator.test_percentage + cfg.synth_generator.validation_percentage) > 1.0:
            print(f'Validation and test percentages must be, together, less than 100%')
            sys.exit(-1)

        # Calcolo quante immagini dovro' creare a partire dal totale:
        # es:   10 immagini totali, 25% test, 15% validazione, allora:
        #       training    : 10 - (10 * 0.25) - (10 * 0.15)
        #       test        : 10 * 0.25
        #       validazione : 10 * 0.15
        self.te_objects = int(cfg.synth_generator.total_objects * cfg.synth_generator.test_percentage)
        self.va_objects = int(cfg.synth_generator.total_objects * cfg.synth_generator.validation_percentage)
        self.tr_objects = cfg.synth_generator.total_objects - self.va_objects - self.te_objects

        # Voglio generare delle immagini di forme, allora mi faro' aiutare
        # da una classe dedicata a fare questo.
        if cfg.synth_generator.type == "shapes":
            self.builder = ShapeBuilder(cfg.shapes)

        self.as_npz = cfg.params.save_as_npz
        self.separate_by_class = cfg.params.separate_by_class
        
        print(f'Images will be created at: {self.out_path}')
        print(f'Images will be stored as {"zip" if self.as_npz else "files"}')
        
        if self.separate_by_class and not self.as_npz:
        
            class_names = self.builder.get_class_names()
                
            for path in [self.tr_path, self.te_path, self.va_path]:
                for name in class_names:
                    sub_path = path / name
                    sub_path.mkdir()

    def build(self) -> None:
        
        # Per creare le immagini sfrutto 3 informazioni:
        # - i prefissi che mettero' ai nomi delle immagini generate.
        # - i percorsi in cui salvarle.
        # - il numero di elementi da generare.
        # Queste informazioni servono per training, test e validazione.
        prefixes = ['tr', 'va', 'te']
        paths = [self.tr_path, self.va_path, self.te_path]
        num_objects = [self.tr_objects, self.va_objects, self.te_objects]
        
        # Con questo ciclo procedo a creare prima training, poi validazione e infine test.
        for prefix, path, objects in zip(prefixes, paths, num_objects):
            print(f'Creating {objects} images at {path}...')
            self.create_synthetic_objects(prefix, path, objects)
            print(f'Images created!')

    def create_synthetic_objects(self, prefix, path, obj_num):
        
        names, images = [], []
        
        # Creo le immagini una alla volta sfruttando l'aiuto del 'builder'             
        for i in range(obj_num):
            
            # ****************************************************************
            # Questo e' il metodo dove a tutti gli effetti si creano le immagini.
            canvas, name = self.builder.create_synthetic_object()
            # ****************************************************************
            
            canvas = self.apply_preprocessing(canvas)
            images.append(canvas)
            if self.separate_by_class:
                names.append(f'{path.as_posix()}/{name}/{prefix}_{i:03}.png')
            else:
                names.append(f'{path.as_posix()}/{prefix}_{i:03}_{name}.png')

        # Infine salvo le immagini zippandole in un .npz o direttamente su disco
        if self.as_npz:
            np.savez(f'{path.as_posix()}/{prefix}.npz', images)
        else:
            for i, n in zip(images, names):
                cv2.imwrite(n, i)

    def apply_preprocessing(self, canvas : np.ndarray) -> np.ndarray:
        # Per semplicita' non applico altri preprocessing.
        # Se avessi voluto applicarli, avrei potuto chiedere l'aiuto
        # della classe 'preprocessor' gia' creata in precedenza.
        return canvas
                
if __name__ == '__main__':

    # Indico dove trovare i json di configurazione e di verifica.
    data_file, schema_file = Path('./config.json'), Path('./config_schema.json')

    print(f'Config file is in path: {data_file}')
    print(f'Config file must follow schema rules from path: {schema_file}')

    # Per proseguire, i percorsi devono essere dei file e i file devono essere json.
    valid_input = data_file.is_file() and schema_file.is_file() and data_file.suffix == '.json' and schema_file.suffix == '.json'
 
    if valid_input:
        
        # Apro i due file e controllo se il json segue lo schema.           
        with open(Path(data_file)) as d:
            with open(Path(schema_file)) as s:

                # Carico i due json e utilizzo lo schema per validare il file di configurazione.
                data, schema = json.load(d), json.load(s)
                
                try:
                    # Il json 'data' e' corretto secondo le regole di 'schema'?
                    jsonschema.validate(instance=data, schema=schema)                    
                except jsonschema.exceptions.ValidationError:
                    print(f'Json config file is not following schema rules.')
                    sys.exit(-1)
                except jsonschema.exceptions.SchemaError:
                    print(f'Json config schema file is invalid.')
                    sys.exit(-1)

    # A questo punto possiedo un file di configurazione json sintatticamente valido.
    # Passo il contenuto al synthetic builder. 
    with open(Path(data_file)) as d:
        
        # Creo un oggetto 'SyntheticBuiled' che si configura seguendo i dati del json.
        sb = SyntheticBuilder(json.loads(d.read(), object_hook=lambda d: SimpleNamespace(**d)))
        sb.build()