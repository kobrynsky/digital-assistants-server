from keras.models import model_from_json
from keras.optimizers import RMSprop
from PIL import Image
import numpy as np
import os

input_size = (224,224, 3)
json_file = open("models/modelResNet.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/modelResNet.h5", by_name=True, skip_mismatch=False)

loaded_model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=2e-4),   
                  metrics=['acc'])

idx_classes_dict = {0: 'Agaricus',
 1: 'Amanita',
 2: 'Boletus',
 3: 'Cortinarius',
 4: 'Entoloma',
 5: 'Hygrocybe',
 6: 'Lactarius',
 7: 'Russula',
 8: 'Suillus'}

def _image_to_array(filename, target_size=input_size):
    img = Image.open(filename)
    img.load()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    x = np.asarray(img, dtype="int32")
    return x

def predict(filepath):
    image = _image_to_array(filepath, input_size[:2])
    image = np.expand_dims(image, axis=0) # macierz o rozmiarze wejściowym sieci
    y = loaded_model.predict(image)
    predictedClass = np.argmax(y, axis=1)
    os.remove(filepath)
    print("Zdjęcie należy do klasy: " + idx_classes_dict.get(predictedClass[0], '19'))
    return (idx_classes_dict.get(predictedClass[0], '19'), max(y[0]))