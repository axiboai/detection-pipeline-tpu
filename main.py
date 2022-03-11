import sys
from ossaudiodev import control_names
import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import object_detector

import tensorflow as tf
from torch import R

#temp until proper logger integrated.
log = print

class model_trainer():
    def __init__(self, labelmap, traindir, testdir, valdir, modelsize):
        '''
        modelsize = small, medium, large (strings)
        '''
        self.labelmap = labelmap
        self.traindir = traindir
        self.testdir = testdir 
        self.valdir = valdir
        self.modelsize = modelsize
        self.dataloaded = False
        # train, test, validate 
        self.data = [[],[],[]]
        
    def load_data(self, label_map, traindir, testdir, valdir):
        """
        - Need: [train, test, validation] x [images, annotations]
        - Each dir should have images folder & annotations folder (pascal format)
        """
        try: 
            train_data = object_detector.DataLoader.from_pascal_voc(
                traindir+'images', traindir+'annotations', label_map=label_map
            )
            validation_data = object_detector.DataLoader.from_pascal_voc(
                testdir+'images', testdir+'annotations', label_map=label_map
            )
            test_data = object_detector.DataLoader.from_pascal_voc(
                valdir+'images', valdir+'annotations', label_map=label_map
            )
            self.dataloaded = True
        except Exception as e:
            log("Unable to load data", e)
            sys.Exit(0)
        self.data[0] = train_data
        self.data[1] = test_data
        self.data[2] = validation_data
        print(f'train count: {len(train_data)}')
        print(f'validation count: {len(validation_data)}')
        print(f'test count: {len(test_data)}')
        

    def train_model(self, train_data, validation_data, test_data, epochs, batch_size):
        '''
        Create a model for training.
        '''
        spec = object_detector.EfficientDetLite3Spec()
        model = object_detector.create(train_data=train_data, 
                                model_spec=spec, 
                                validation_data=validation_data, 
                                epochs=epochs, 
                                batch_size=batch_size, 
                                train_whole_model=True)
        model.evaluate(test_data)
        return model

    def save_model(dir):
        '''
        Save the model.
        '''
        pass 

class inference_engine():
    def __init__(self, modeldir, size):
        
    
def main():
    label_map = {1: "facerotation"}
    trainer = model_trainer(
        label_map, "../dataset/train/", "../dataset/test/", "../dataset/validation/", "small"
    )
    

if __name__ == "__main__":
    main()
    