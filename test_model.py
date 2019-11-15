from model import *

if __name__ == '__main__':
    # TODO: Available GPU check for keras & tf backend
    # from keras import backend
    # if len(backend.tensorflow_backend._get_available_gpus()) == 0:
    #     print('WARNING: No available GPUs found by Keras')
    m = Model()
    m.train()
    # m.test('10_epochs.h5', 'data/train/bishop/20191108_142519.jpg')
