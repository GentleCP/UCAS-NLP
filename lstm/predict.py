# predict
from keras.models import load_model
import pickle

model = load_model('model/lstm.h5')
tok = pickle.load(open('model/tok.pickle', 'rb').read())