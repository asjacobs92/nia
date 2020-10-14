############ DATASET ############
DATASET_PATH = '../res/dataset.txt'
DATASET_SIZE = 5000
DATASET_VOCAB_SIZE = 2000
DATASET_FILTERS = '!$&*+,-./;<=>?#[\\]^_`{|}~\t\n'

DATASET_HOURS = ['', '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                 '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
DATASET_TARGETS = ['', 'att', 'bell', 'ufrgs', 'pucrs', 'company', 'adp', 'example']
DATASET_MIDDLEBOXES = ['firewall', 'dpi', 'ids', 'load-balancer', 'parental-control']
DATASET_USERNAMES = ['asjacobs', 'granville', 'rjpfitscher', 'ronaldo']
DATASET_LOCATIONS = ['', 'database', 'marketing', 'gateway', 'backend', 'ps4', 'asjacobs-pc']
DATASET_TRAFFIC = ['', 'udp', 'http', 'netflix', 'youtube']
DATASET_QOS_METRICS = [['latency', 'ms'], ['loss', '%'], ['jitter', 'ms'], ['bandwidth', 'mbps']]
DATASET_QOS_CONSTRAINTS = ['less', 'more', 'equal', 'none']

############ MODEL ############
MODEL_DIR = '../res/weights/'
MODEL_TEST_INPUT_PATH = '../res/test_input.txt'
MODEL_TEST_RESULT_PATH = '../res/test_result.txt'
MODEL_TRAINED_PATH = '../res/weights/model.h5'
MODEL_WEIGHTS_PATH = '../res/weights/model_weights_{}.hdf5'

MODEL_ACTIVATION = 'softmax'
MODEL_OPTIMIZER = 'rmsprop'
MODEL_METRICS = ['accuracy']
MODEL_LOSS = 'categorical_crossentropy'

MODEL_BATCH_SIZE = 64         # Batch size for training.
MODEL_EPOCHS = 100            # Number of epochs to train for.
MODEL_LATENT_DIM = 256        # Latent dimensionality of the encoding space.
MODEL_HIDDEN_DIM = 1000       # Dimensionality of hidden layer.
MODEL_HIDDEN_LAYERS = 3       # Dimensionality of hidden layer.
MODEL_VALIDATION_SPLIT = 0.2  # Fraction of dataset used to validate model fitting.
