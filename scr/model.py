import numpy as np
import sys
from itertools import chain
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Définition de votre méthode améliorée
class ModelBasedClassImproved:
    
    def __init__(self, Lambda = 0.5):
        self.Lambda = Lambda
        self.class_intervals = {}
        
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        # Entraîner une forêt aléatoire pour calculer l'importance des caractéristiques
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        self.importances = rf.feature_importances_
        
        #Calculate IQR Limit Interval
        self.Calculate_class_intervals()
            
        #Calculate Mean Model
        self.mean_Model(X, y)
        
    def Calculate_class_intervals(self):
        
        X = self.X_train
        y = self.y_train
        self.classes = np.unique(y)
        self.features = X.shape[1]
        
        for cls in self.classes:
            cls_indices = np.where(y == cls)[0]
            cls_data = X[cls_indices]
            
            class_intervals = []
            
            for feature in cls_data.T:
                Q1 = np.percentile(feature, 25)
                Q3 = np.percentile(feature, 75)
                IQR = Q3 - Q1
                
                borne_inferieure = Q1 - 1.5 * IQR
                borne_superieure = Q3 + 1.5 * IQR
                
                filtered_feature = feature[(feature >= borne_inferieure) & (feature <= borne_superieure)]
                
                feature_min = filtered_feature.min()
                feature_max = filtered_feature.max()
                
                class_intervals.append((feature_min, feature_max))
            
            self.class_intervals[cls] = class_intervals

    
    def mean_Model(self, X, y):
        # Initialiser les dictionnaires pour stocker les sommes et les comptages
        sums = {}
        counts = {}
        self.means = {}
        
        # Initialiser les dictionnaires pour chaque classe
        for class_label in np.unique(y):
            sums[class_label] = np.zeros(X.shape[1])
            counts[class_label] = 0
        
        # Calculer les sommes et les comptages pour chaque classe
        for i, label in enumerate(y):
            sums[label] += X[i]
            counts[label] += 1
        
        # Calculer les moyennes pour chaque classe
        for class_label in sums:
            self.means[class_label] = sums[class_label] / counts[class_label]    
    

    def normalize(self, values):
        total_sum = np.sum(values)
        return values / total_sum if total_sum != 0 else values


    def _membership_degree(self, x, clss):
        degree = 0
        for i in range(self.features):
            min_val, max_val = self.class_intervals[clss][i]
            mid_val = (min_val + max_val) / 2
            if min_val <= x[i] <= max_val:
                if x[i] == mid_val:
                    degree += (7 * self.importances[i])
                else:
                    degree += ((7 - abs(mid_val - x[i]) / (max_val - min_val)) * self.importances[i])
            
        return degree

    def _predict(self, x):
        # Calculer les distances pondérées entre x et les moyennes
        distances = [self._distance(x, mean) for class_label, mean in self.means.items()]
        degrees = [self._membership_degree(x, clss) for clss in self.classes]
        
        degDis = [self.Lambda * dis + ((1-self.Lambda)/deg) for dis, deg in zip(distances, degrees)]
        
        closest_class = degDis.index(min(degDis))
        distances = self.normalize(distances)
        
        return closest_class

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _distance(self, x1, x2):
        return np.sqrt(np.sum(self.importances * ((x1 - x2) ** 2)))

#-----------------------------------------------------------------------------------------

# Fonction pour calculer la taille totale d'un objet en Mo (y compris les objets imbriqués)
def total_size(o, handlers={}, verbose=False):
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter}
    all_handlers.update(handlers)
    seen = set()
    
    def sizeof(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o)
        if verbose:
            print(s, type(o), repr(o))
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s
    
    # Retourner la taille en Mo
    return sizeof(o) / (1024 ** 2)


# KNN classique sans scikit-learn
def knn_classic(X_train, y_train, X_test, k=3):
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict_instance(x):
        distances = [euclidean_distance(x, X_train[i]) for i in range(len(X_train))]
        neighbors = np.argsort(distances)[:k]
        neighbor_labels = [y_train[i] for i in neighbors]
        return np.argmax(np.bincount(neighbor_labels))

    start_train_time = time.time()
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    start_pred_time = time.time()
    predictions = np.array([predict_instance(x) for x in X_test])
    end_pred_time = time.time()
    pred_time = end_pred_time - start_pred_time

    total_time = train_time + pred_time
    return predictions, train_time, pred_time, total_time

# KNN avec scikit-learn
def knn_sklearn(X_train, y_train, X_test, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    start_train_time = time.time()
    knn.fit(X_train, y_train)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    start_pred_time = time.time()
    predictions = knn.predict(X_test)
    end_pred_time = time.time()
    pred_time = end_pred_time - start_pred_time

    total_time = train_time + pred_time
    return predictions, train_time, pred_time, total_time
    