import numpy as np

class MeanStack:
    
    def __init__(self, models):
        self.models = models
    
    def fit(self, X_train, y_train):
        
        for model in self.models:
            t0 = datetime.datetime.now()
            print('Trainig:', type(model).__name__)
            print('X_train.shape', X_train.shape)
            print('y_train.shape', y_train.shape)
            model.fit(X_train, y_train)
            
            print(type(model).__name__, ' training time:', datetime.datetime.now() - t0)
            
            
    
    def predict(self, X):
        
        preds = [model.predict(X) for model in self.models]
        preds_mean = np.mean(preds, axis=0)
        preds_result = preds_mean >= 0.5
        
        return np.array(preds_result >= 0.5, dtype='int32')
   