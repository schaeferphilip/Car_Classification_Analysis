import configparser
import pickle

class maclass : 
    def __init__(self, config_path) -> None:
        
        config = configparser.ConfigParser()
        config.read(config_path)
        PATH_DATA_PREPROCESSOR = config.get('Paths', 'PATH_DATA_PREPROCESSOR')
        PATH_MODEL = config.get('Paths', 'PATH_MODEL')
        PATH_VEHICLE_CATEGORIES = config.get('Paths', 'PATH_VEHICLE_CATEGORIES')

        # Load data preprocessor
        with open(PATH_DATA_PREPROCESSOR, 'rb') as file:
            self.data_preprocessor = pickle.load(file)

        # Load model
        with open(PATH_MODEL, 'rb') as file:
            self.model = pickle.load(file)

        # Load vehicles by categorie
        with open(PATH_VEHICLE_CATEGORIES, 'rb') as file:
            self.vehicle_cats = pickle.load(file)
        
        self.features = []
        self.target_name = ["véhicules normaux", "vieux véhicules", "véhicules familiaux", "citadines", "véhicules luxueux "]

    def predict(self, X):
        X = self.preprocess(X)
        cat_vehicle = self.model.predict(X)[0]
        return self.postprocess(cat_vehicle)
    
    def preprocess(self, X):
        X = self.data_preprocessor.transform(X)
        return X

    def postprocess(self, category): 
        return self.target_name[category], self.vehicle_cats[category].drop(columns=['label'])