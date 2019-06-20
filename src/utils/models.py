import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape, Add, Multiply, Subtract, Dropout
from keras.layers import Conv2D, MaxPooling2D, LocallyConnected1D, Conv1D, UpSampling1D, MaxPooling1D, Dot, Concatenate, LSTM, TimeDistributed

def conv1D_lon(idx_sensor, n_sensors=16):
    ''' Returns a model using all the sensors to predict index_sensor '''
    xin = Input(shape=(n_sensors, 1), name='main_input')
    x = LocallyConnected1D(8, 7, data_format = 'channels_last', padding='valid')(xin)
    x = Activation('relu')(x)
    x = LocallyConnected1D(16, 5, data_format = 'channels_last', padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv1D(32, 3, data_format = 'channels_last', padding='causal')(x)
    xl = Flatten()(x)
    xl = Dropout(0.2)(xl)
    xo = Dense(1)(xl)

    # use date info here?
    xinf = Flatten()(xin)
    s  = Dense(5)(xinf)
    s = Activation('tanh')(s)
    s = Dense(2)(s)
    s = Activation('softmax')(s)

    # sort of residual connection
    xin_0 = Activation('relu')(xin)
    xin_1 = Lambda(lambda x : x[:, idx_sensor, :])(xin_0)
    xo_m = Dot(axes=1)([Concatenate()([xo, xin_1]), s])
    xo_m = Activation('relu')(xo_m)

    model = Model(inputs=[xin], outputs=[xo_m])
    return model


def conv1D_lon_lat(idx_sensor, n_sensors=16):
    ''' Returns a model using all the sensors to predict index_sensor '''
    xin = Input(shape=(n_sensors, 1), name='lon_input')
    x = LocallyConnected1D(8, 7, data_format = 'channels_last', padding='valid')(xin)
    x = Activation('relu')(x)
    x = LocallyConnected1D(16, 5, data_format = 'channels_last', padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv1D(32, 3, data_format = 'channels_last', padding='causal')(x)
    xl = Flatten()(x) 

    yin = Input(shape=(n_sensors, 1), name='lat_input')
    y = LocallyConnected1D(8, 7, data_format = 'channels_last', padding='valid')(xin)
    y = Activation('relu')(x)
    y = LocallyConnected1D(16, 5, data_format = 'channels_last', padding='valid')(x)
    y = Activation('relu')(x)
    y = Conv1D(32, 3, data_format = 'channels_last', padding='causal')(x)
    yl = Flatten()(y)

    xc = Concatenate()([xl, yl])
    xc = Dropout(0.2)(xc)
    xo = Dense(1)(xc)

    # use date info here?
    xinf = Flatten()(xin)
    s  = Dense(5)(xinf)
    s = Activation('tanh')(s)
    s = Dense(2)(s)
    s = Activation('softmax')(s)

    # sort of residual connection
    xin_0 = Activation('relu')(xin)
    xin_1 = Lambda(lambda x : x[:, idx_sensor, :])(xin_0)
    xo_m = Dot(axes=1)([Concatenate()([xo, xin_1]), s])
    xo_m = Activation('relu')(xo_m)

    model = Model(inputs=[xin, yin], outputs=[xo_m])
    return model


def conv1D_lon_LSTM(n_steps=3, n_sensors=16):
    ''' Returns a model using all the sensors to predict index_sensor '''
    xin = Input(shape=(n_steps, n_sensors, 1), name='main_input')
    x = TimeDistributed(
            LocallyConnected1D(8, 7,  data_format = 'channels_last', padding='valid', activation='relu')
        )(xin)
    x = TimeDistributed(
            LocallyConnected1D(16, 5, data_format = 'channels_last', padding='valid', activation='relu')
        )(x)
    x = TimeDistributed(
            Conv1D(32, 3, data_format = 'channels_last', padding='causal')
        )(x)
    xl = TimeDistributed(
            Flatten()
        )(x)
    xl = LSTM(20)(xl)
    xl = Dropout(0.2)(xl)
    xo = Dense(n_sensors)(xl)

    model = Model(inputs=[xin], outputs=[xo])
    return model
