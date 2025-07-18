import tensorflow as tf
from Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV

#Used to optimize the hyperparameter
class Param_Op:
    def __init__(self,dnn, param, X_train, y_train):
        self.model = RandomizedSearchCV(dnn, param_distributions=param, cv=5)
        self.model.fit(X_train, y_train)
        dnn_opt = self.model.best_estimator_
        print("Parameter: ")
        print(dnn_opt)

class ANN(Classifier):
    def __init__(self, data_summary):
        super().__init__()

        #gpus = tf.config.experimental.list_physical_devices('GPU')
        #print("List of physical devices: {}".format(gpus))
        #tf.config.get_visible_devices()

        #construct a fully connected NN
        self.model = tf.keras.models.Sequential()

        #TODO normalize, zero to one, the predicted BW
        #Input layer
        #self.input_shape = 2 * nodes + 1 + links
        self.input_shape = data_summary['n_node_src'] + data_summary['n_node_dst'] + 1 + data_summary['n_link']
        print('Input size: {}'.format(self.input_shape))
        #hidden layer 1
        #self.model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(self.input_shape,)))
        self.model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(self.input_shape,)))
        self.model.add(tf.keras.layers.Dropout(0.2))

        # hidden layer 2
        #self.model.add(tf.keras.layers.Dense(units=100, activation='relu'))
        #self.model.add(tf.keras.layers.Dropout(0.2))

        #Output layer
        #self.output_shape = links
        self.output_shape = data_summary['n_path']
        self.model.add(tf.keras.layers.Dense(units=self.output_shape, activation='sigmoid'))

        print(data_summary['n_node_src'])
        print(data_summary['n_node_dst'])
        print(data_summary['n_link'])
        print(data_summary['n_path'])
        #opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
        #opt = tf.keras.optimizers.Adam(learning_rate=0.00000001)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_crossentropy', tf.keras.metrics.Precision()])
        self.summary()

    def train(self, train_data, train_targets, val_data, val_targets, save=False, file_name='parameter.h5'):

        #with tf.device("/device:GPU:0"):
        #history = self.model.fit(train_data, train_targets, validation_data=(val_data, val_targets), batch_size=10, epochs=10, verbose=1)
        history = self.model.fit(train_data, train_targets, batch_size=10, epochs=5, verbose=1)
        if save:
            self.model.save(file_name)
