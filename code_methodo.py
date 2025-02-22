# Prep dataset with strafication on strains/specimens/etc: train & test set (x_train, x_test, y_train, y_test)

from libraries_utils import *
from functions_utils_for_nn import *

len_sp = 17999


class Methodology_models():
    
    def __init__(self, model_name, len_sp, nb_class, block_size=None, spectrum_length=None):
        self.model_name = model_name
        
        self.X_train = None
        self.y_train = None 
        self.X_va = None
        self.y_va = None
        
        self.X_test = None
        self.y_test = None
        
        self.len_sp = len_sp
        self.nb_class = nb_class
        self.model = None
        
        self.block_size = block_size
        self.spectrum_length = spectrum_length
        
        if block_size :
            assert spectrum_length/block_size == spectrum_length//block_size
            
        
        # Choose model
        if self.model_name == 'cnn_1d':
            self.model = self.cnn_1d_uncompiled_model(nb_shape = self.len_sp)
        
        elif self.model_name == 'tcn_1d':
            self.model = self.tcn_1d_uncompiled_model(nb_shape = self.len_sp)

        elif self.model_name == 'cnn_1d_anopheles_age':
            self.model = self.cnn_1d_anopheles_age_uncompiled_model(nb_shape = self.len_sp)

        elif self.model_name == 'tcn_1d_anopheles_age':
            self.model = self.tcn_1d_anopheles_age_uncompiled_model(nb_shape = self.len_sp)
            
        elif self.model_name == 'rec_1d':    
            self.model = self.rec_1d_uncompiled_model()
            
        elif self.model_name == 'esn_1d':    
            self.model = self.esn_1d_uncompiled_model()
        
        #elif #other models
        
        else:
            raise ValueError("Model is currently not support, please use 'cnn_1d' instead.")
    
    
    def data_preparation_zero_shot_train(self, data_train, len_sp, feature_of_stratification, target, colonne, encode=True, n_splits_k_fold = 5, random_state_val = 1234):
    
        data_ML = data_train.reset_index(drop=True)

        
        encoder = LabelEncoder()
        encoder.fit(list(data_ML[feature_of_stratification]))
        data_ML[feature_of_stratification] = encoder.transform(data_ML[feature_of_stratification])
        
        if encode :
            encoder.fit(list(data_ML[target]))
            data_ML[target] = encoder.transform(data_ML[target])
        else : 
            print('Non encode target')


        train = data_ML[[feature_of_stratification, colonne , target]]
        y_train = data_ML[[feature_of_stratification, target]]
        train_label = []
        y_train_label = []
        for classe in np.unique(train[target]):
            train_label.append(np.unique(train[train[target]==classe][feature_of_stratification]))
            y_train_label.append(np.repeat(classe, repeats=len(np.unique(train[train[target]==classe][feature_of_stratification]))))
        train_label = np.concatenate(train_label).ravel()
        y_train_label = np.concatenate(y_train_label).ravel()

        ssplit=StratifiedKFold(n_splits = n_splits_k_fold ,shuffle = True, random_state = random_state_val)
        for train_idx, va_idx in ssplit.split(train_label, y_train_label):
            train_index = train_idx
            va_index = va_idx     #train and test strains/specimens are separated beforehand 
            #at each fold, the strains/specimens of the validation set are totally different, so train and validation are never the same on each fold

        X_tr_lab = [train_label[i] for i in train_index]
        y_tr_lab= [y_train_label[i] for i in train_index]
        X_va_lab = [train_label[i] for i in va_index]
        y_va_lab = [y_train_label[i] for i in va_index]

        print("Train label : ", X_tr_lab, "size : ", len(X_tr_lab))
        print("Y Train clone/non clone : ", y_tr_lab)

        print("Val label : ", X_va_lab, "size : ", len(X_va_lab))
        print("Y val clone/non clone : ", y_va_lab)
        
        X_train = np.array([spectre[:len_sp] for spectre in train[train[feature_of_stratification].isin(X_tr_lab)][colonne].values])
        
        if self.model_name in ['rec_1d', 'esn_1d'] :
            X_train = np.expand_dims(X_train, axis=-1)
        
        y_train = train[train[feature_of_stratification].isin(X_tr_lab)][target]
        y_train = np.array(y_train.values.astype('float32').reshape(-1,1))
        print(f'Train data dimensions: {X_train.shape}, {y_train.shape}')

        X_va = np.array([spectre[:len_sp] for spectre in train[train[feature_of_stratification].isin(X_va_lab)][colonne].values])
        
        if self.model_name in ['rec_1d', 'esn_1d'] :
            X_va = np.expand_dims(X_va, axis=-1)
        
        y_va = train[train[feature_of_stratification].isin(X_va_lab)][target]
        y_va = np.array(y_va.values.astype('float32').reshape(-1,1))
        print(f'Validation data dimensions: {X_va.shape}, {y_va.shape}')
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_va = X_va
        self.y_va = y_va

        return self.X_train, self.y_train, self.X_va, self.y_va
    
    
    
    def data_preparation_zero_shot_test(self, data_test, len_sp, feature_of_stratification, target,colonne, encode=True):
    
        data_ML_test = data_test.reset_index(drop=True)

        encoder = preprocessing.LabelEncoder()

        encoder.fit(list(data_ML_test[feature_of_stratification]))
        data_ML_test[feature_of_stratification] = encoder.transform(data_ML_test[feature_of_stratification])
        
        if encode :
            encoder.fit(list(data_ML_test[target]))
            data_ML_test[target] = encoder.transform(data_ML_test[target])
        else : 
            print('Non encode target')

        test = data_ML_test[[feature_of_stratification, colonne ,  target]]
        y_test = data_ML_test[[feature_of_stratification, target]]
        test_label = []
        y_test_label = []
        for classe in np.unique(test[target]):
            test_label.append(np.unique(test[test[target]==classe][feature_of_stratification]))
            y_test_label.append(np.repeat(classe, repeats=len(np.unique(test[test[target]==classe][feature_of_stratification]))))
        test_label = np.concatenate(test_label).ravel()
        y_test_label = np.concatenate(y_test_label).ravel()

        print("Test label : ", test_label, "size : ", len(test_label))
        print("Y test clone/non clone : ", y_test_label)

        #test_data = np.array([np.asarray(spectre[:len_sp]).astype('float32') for spectre in test[colonne].values])
        #y_test_data = test[target]

        X_test = np.array([np.asarray(spectre[:len_sp]).astype('float32') for spectre in test[colonne].values])
        X_test = np.stack(X_test)
        
        if self.model_name in ['rec_1d', 'esn_1d'] :
            X_test = np.expand_dims(X_test, axis=-1)
            X_test.resize(len(X_test), self.spectrum_length//self.block_size, self.block_size)
        
        y_test = test[target]
        y_test = np.array(y_test.values.astype('float32').reshape(-1,1))
        print(f'Test data dimensions: {X_test.shape}, {y_test.shape}')
        
        self.X_test = X_test
        self.y_test = y_test 

        return self.X_test, self.y_test 
    
    
    def cnn_1d_uncompiled_model(self, nb_shape):
        inputs = Input(shape=(nb_shape, 1)) #X_train.shape[1]
        x = Conv1D(3, 6, activation='relu')(inputs)
        x = MaxPool1D(pool_size=100)(x)
        x = Flatten()(x)

        x = Dense(1024, activation='relu')(x) 

        x = LayerNormalization()(x)
        
        if self.nb_class >=2:
            activation_output = 'softmax'
        elif self.nb_class == 1:
            activation_output = 'relu'
        else :
            raise ValueError("Number of class for uncompiled model function is unclear.")
            
        predictions = Dense(self.nb_class, activation = activation_output)(x)

        model_cnn_1d = Model(inputs=[inputs], outputs=predictions)
        
        return model_cnn_1d
    
    def cnn_1d_anopheles_age_uncompiled_model(self, nb_shape):
        inputs = Input(shape=(nb_shape, 1)) #X_train.shape[1]
        x = Conv1D(3, 6, activation='relu')(inputs)
        x = MaxPool1D(pool_size=100)(x)
        x = Flatten()(x)

        x = Dense(512, activation='relu')(x) 
        x = Dense(512, activation='relu')(x)

        x = LayerNormalization()(x)
        
        if self.nb_class >=2:
            activation_output = 'softmax'
        elif self.nb_class == 1:
            activation_output = 'relu'
        else :
            raise ValueError("Number of class for uncompiled model function is unclear.")
            
        predictions = Dense(self.nb_class, activation = activation_output)(x)

        model_cnn_1d_anopheles_age = Model(inputs=[inputs], outputs=predictions)
        
        return model_cnn_1d_anopheles_age
    
    
    
    def tcn_1d_uncompiled_model(self, nb_shape):
        inputs = Input(shape=(nb_shape, 1)) #X_train.shape[1]
        x = SpatialDropout1D(0.3)(inputs)
        x = TCN(nb_filters=6, kernel_size=7, nb_stacks=1, dilations=[3], padding='same',use_skip_connections=True,
                dropout_rate=0.3, return_sequences=True)(x)
        x = MaxPool1D(pool_size=100)(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        #x = Dense(512, activation='relu')(x)
        x = LayerNormalization()(x)
        
        if self.nb_class >=2:
            activation_output = 'softmax'
        elif self.nb_class == 1:
            activation_output = 'relu'
        else :
            raise ValueError("Number of class for uncompiled model function is unclear.")
            
        predictions = Dense(self.nb_class, activation = activation_output)(x)

        model_tcn_1d = Model(inputs=[inputs], outputs=predictions)
        
        return model_tcn_1d
    

    def tcn_1d_anopheles_age_uncompiled_model(self, nb_shape):
        inputs = Input(shape=(nb_shape, 1)) #X_train.shape[1]
        x = BatchNormalization(momentum=0.99)(inputs)
        x = SpatialDropout1D(0.3)(x)
        x = TCN(nb_filters=3, kernel_size=10, nb_stacks=1, dilations=[10], padding='same',use_skip_connections=True,
                dropout_rate=0.3, return_sequences=True)(x)
        x = MaxPool1D(pool_size=50)(x)
        x = Flatten()(x)
        
        
        x = Dense(1024,activation=tf.keras.layers.PReLU(), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x) #essayer 512 ou 2048
        x = Dense(512,activation=tf.keras.layers.PReLU(), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = Dense(1024,activation=tf.keras.layers.PReLU(), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        
        x = LayerNormalization()(x)
        
        if self.nb_class >=2:
            activation_output = 'softmax'
        elif self.nb_class == 1:
            activation_output = 'relu'
        else :
            raise ValueError("Number of class for uncompiled model function is unclear.")
            
        predictions = Dense(self.nb_class, activation = activation_output)(x)

        model_tcn_1d_anopheles_age = Model(inputs=[inputs], outputs=predictions)
        
        return model_tcn_1d_anopheles_age
    
    
    
    def rec_1d_uncompiled_model(self):
        
        number_of_blocks = self.spectrum_length//self.block_size
        
        inputs = Input(shape=(number_of_blocks, self.block_size))
        x = BatchNormalization(momentum=0.98)(inputs)
        x = Bidirectional(GRU(250,dropout=0.5, recurrent_dropout=0.5, activation=tf.keras.layers.LeakyReLU(alpha=0.1), return_sequences=True))(x)
        #x = Bidirectional(GRU(250,dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True))(x)
        x = Attention(number_of_blocks)(x)
        x = Dense(512,activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)#'relu',  tf.keras.layers.PReLU()
        x = Dropout(.3)(x)
        #x = Dense(256,activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x) #tf.keras.layers.LeakyReLU(alpha=0.1)
        #x = Dense(1024,activation='relu')(x)
        x = LayerNormalization()(x)
        
        if self.nb_class >=2:
            activation_output = 'softmax'
        elif self.nb_class == 1: ################# specify if regression for age estimation or binary classif
            activation_output = 'relu' #sigmoid
        else :
            raise ValueError("Number of class for uncompiled model function is unclear.")
            
        predictions = Dense(self.nb_class, activation = activation_output)(x)

        model_rec_1d = Model(inputs=[inputs], outputs=predictions)
        
        return model_rec_1d
    
    
    
    def esn_1d_uncompiled_model(self):
        
        number_of_blocks = self.spectrum_length//self.block_size
        
        inputs = Input(shape=(number_of_blocks, self.block_size))
        x = BatchNormalization(momentum=0.98)(inputs)
        x = ESN(units=250, activation='tanh', return_sequences=True)(x)
        x = Attention(number_of_blocks)(x)
        x = Dense(512,activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
        x = LayerNormalization()(x)
        
        if self.nb_class >=2: 
            activation_output = 'softmax'
        elif self.nb_class == 1: ################# specify if regression for age estimation or binary classif
            activation_output = 'relu' #sigmoid
        else :
            raise ValueError("Number of class for uncompiled model function is unclear.")
            
        predictions = Dense(self.nb_class, activation = activation_output)(x)

        model_esn_1d = Model(inputs=[inputs], outputs=predictions)
        
        return model_esn_1d
    
    
    
    def model_fit_compile(self, X_train, y_train, X_va, y_va, BATCH, EPOCH, types, set_class_weight = False):
        
        model = self.model
    
        if self.nb_class >1:
            y_train = to_categorical(y_train, num_classes= self.nb_class)
            y_va = to_categorical(y_va, num_classes= self.nb_class)
        elif self.nb_class == 1:
            pass
        else:
            raise ValueError('Type de prédiction non reconnue : indiquez le nombre de sortie (>=1)')

        ##########################################################

        if self.nb_class == 1:
            if types == 'classification' :
                loss_for_nn = "binary_crossentropy"
                metric_for_nn = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(name='precision')]
                metric_name = 'binary_accuracy'
            elif types == 'regression' :
                loss_for_nn = 'huber_loss'
                metric_for_nn = ['mae']
                metric_name = 'mae'
            else :
                raise ValueError('types for nb_class == 1 for metrics and loss is UNKNOWN')
            
        elif self.nb_class == 2:
            metric_for_nn = [tf.keras.metrics.CategoricalAccuracy(), f1_m] #['acc', f1_m]
            metric_name = 'categorical_accuracy' #'acc'
            loss_for_nn = 'categorical_crossentropy' 
        elif self.nb_class >= 3:
            metric_for_nn = [tf.keras.metrics.CategoricalAccuracy(), f1_m]
            metric_name = 'categorical_accuracy'
            loss_for_nn = 'categorical_crossentropy'
        else :
            raise ValueError('Indiquez le nombre de sortie (>=1) ou revoir les paramètres de compilation du modèle')

        ########################################################### 

        model.compile(optimizer='adam', loss=loss_for_nn, metrics=metric_for_nn) #AUC or f1 score

        print(model.summary())


        # Define a learning rate decay method:
        #lr_decay = ReduceLROnPlateau(monitor='loss', 
        #                            patience=1, verbose=0, 
        #                            factor=0.5, min_lr=1e-8)

        # Define Early Stopping:
        early_stop = EarlyStopping(monitor='val_'+'{}'.format(metric_name), min_delta=0, 
                                patience=20, verbose=1, mode='auto',
                                baseline=0, restore_best_weights=True)
        #reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, min_lr=1e-8)

        if set_class_weight==True :
            class_weights = generate_class_weights(np.argmax(y_train, axis=1),multi_class=True, one_hot_encoded=False)
            print('class weights : ', class_weights)
        else :
            class_weights = None



        # fit model
        
        if self.model_name in ['rec_1d', 'esn_1d'] :
            
            Steps_per_epoch_train = len(X_train)//BATCH 
            Steps_per_epoch_val = len(X_va)//BATCH
            
            History = model.fit(new_my_batch_generator_timestep(X_train, y_train, BATCH, self.block_size, self.spectrum_length), 
                        validation_data=new_my_batch_generator_timestep(X_va, y_va, BATCH, self.block_size, self.spectrum_length),
                    epochs=EPOCH, batch_size = BATCH, steps_per_epoch=Steps_per_epoch_train, shuffle=True, callbacks = [early_stop],
                    verbose=1, workers=1,use_multiprocessing=False, validation_steps = Steps_per_epoch_val, class_weight = class_weights)
            
            results = model.evaluate(new_my_batch_generator_timestep(X_va, y_va, BATCH, self.block_size, self.spectrum_length), verbose = 1, 
                            batch_size = BATCH, steps=7)
            
        else : 
            History = model.fit(X_train, y_train,
                          epochs=EPOCH,
                          batch_size=BATCH,
                          validation_data=(X_va, y_va),
                          shuffle=True,verbose=1,
                          callbacks=[early_stop], class_weight = class_weights)
            
            results = model.evaluate(X_va,  y_va, verbose = 1)

        
        history_dict=History.history
        loss_values = history_dict['loss']
        val_loss_values=history_dict['val_loss']
        plt.plot(loss_values,'bo',label='training_loss')
        plt.plot(val_loss_values,'r',label='val loss')
        plt.legend(loc = 'upper right')

        plt.show()

        
        results = dict(zip(model.metrics_names,results))

        #VALIDATION_ACCURACY.append(results[metric_name])
        #VALIDATION_LOSS.append(results['loss'])

        #if self.nb_class >1:
        #    VALIDATION_F1_SCORE.append(results['f1_m'])
        
        
        self.model = model

        return model


    
    def model_predict(self, X_test, y_test, types):
        model = self.model

        y_test_pred = model.predict(X_test)

        if types=='classification':
            if self.nb_class > 1:
                y_test_pred = np.argmax(y_test_pred, axis=1)
            elif self.nb_class ==1:
                y_test_pred = (y_test_pred > 0.5).astype(np.int)
                y_test_pred = np.array(y_test_pred).ravel()
            else :
                raise ValueError('Treatment for y_test_pred for classification types is UNKNOWN.')

        elif types=='regression':
            y_test_pred = y_test_pred.ravel()

        else :
            raise ValueError('Type non reconnu')

        print("pred : ", repr(y_test.ravel()))
        print("test pred : ", repr(y_test_pred))

        return y_test_pred    
    
    
    
    def clean_model_session(self):       
        tf.keras.backend.clear_session()
        gc.collect()
        tf.random.set_seed(42)
        
        del self.model