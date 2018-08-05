import os.path
import pickle

import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Activation, Concatenate, Conv2D, Dense, Dropout,
                          Flatten, Input, concatenate)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from tqdm import tqdm


# ビットボードをnp.array([8,8])に変換
def bitboard_convert_to_nparray(bitb):
    mask = np.uint64(0x8000000000000000)
    na = np.zeros((64), dtype='int')
    if bitb == 0:
        return np.zeros((8, 8), dtype='int')
    for i in range(64):
        if bitb & mask:  # maskedにビットが立ってる
            na[i] = 1
            bitb &= ~mask
            if not bitb:  # bitbがFlase->立っているビットがないのでreturn
                return na.reshape(8, 8)
        mask >>= np.uint64(1)


def readkifu(kifu_list_file):
    print(f'{kifu_list_file}を読み込んでいます')
    bl = []  # ban_list
    ml = []  # move_list
    rl = []  # result_list
    bls = np.empty((0, 3, 8, 8))
    mls = np.empty((0, 65))
    rls = np.empty((0, 2))
    cnt = 1
    with open(kifu_list_file, 'r')as f:
        for line in tqdm(f.readlines()):
            kifu = line.rstrip('\r\n')
            for i in np.load(kifu):
                i = i.astype(np.uint64)
                bb = bitboard_convert_to_nparray(i[0])  # black_board
                wb = bitboard_convert_to_nparray(i[1])  # white_board
                teban = np.full((8, 8), i[2], dtype='int')
                ban = np.stack([bb, wb, teban])
                move = np.eye(65, dtype='int')[i[3]]
                result = np.eye(2, dtype='int')[i[4]]
                bl.append(ban)
                ml.append(move)
                rl.append(result)
            cnt += 1
            if cnt % 1000 == 0:
                bls = np.append(bls, np.array(bl), axis=0)
                mls = np.append(mls, np.array(ml), axis=0)
                rls = np.append(rls, np.array(rl), axis=0)
                bl = []
                ml = []
                rl = []
        bls = np.append(bls, np.array(bl), axis=0)
        mls = np.append(mls, np.array(ml), axis=0)
        rls = np.append(rls, np.array(rl), axis=0)
        bl = []
        ml = []
        rl = []
        print(f'{kifu_list_file}を読み込みました')
    return bls, mls, rls

if __name__ == '__main__':
    b_train, m_train, r_train = readkifu('kifulist_train.txt')
    b_test, m_test, r_test = readkifu('kifulist_test.txt')
    if os.path.isfile('model_policy_value.h5'):
        model = load_model('model_policy_value.h5')
        print('モデルmodel_policy_value.h5を読み込みました')
    else:
        print('モデルはありません')
        ban_input = Input(shape=(3, 8, 8))
        c = Conv2D(192, (3, 3),
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.001),
                   activation='relu')(ban_input)
        d = Dropout(0, 2)(c)
        c = Conv2D(192, (3, 3),
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.001),
                   activation='relu')(d)
        d = Dropout(0, 2)(c)
        c = Conv2D(192, (3, 3),
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.001),
                   activation='relu')(d)
        d = Dropout(0, 2)(c)
        x = Flatten()(d)
        dens_1 = Dense(64, kernel_regularizer=regularizers.l2(0.001),
                       activation='relu')(x)
        dens_2 = Dense(64, kernel_regularizer=regularizers.l2(0.001),
                       activation='relu')(dens_1)
        d = Dropout(0.5)(dens_2)
        move_output = Dense(65, activation='softmax', name='move')(d)
        result_output = Dense(2, activation='softmax', name='result')(d)
        model = Model(inputs=ban_input,
                      outputs=[move_output, result_output])
    model.summary()
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=1)
    md_ch = ModelCheckpoint(filepath='model_policy_value.h5')
    history = model.fit(b_train, [m_train, r_train],
                        epochs=20, batch_size=32,
                        verbose=1, validation_data=(b_test, [m_test, r_test]),
                        callbacks=[early_stopping, md_ch])
    model.save('model_policy_value.h5')
    pred = model.predict(b_test[-100:], batch_size=1, verbose=1)
    pred_move = pred[0].argmax(axis=1)
    pred_result = pred[1].argmax(axis=1)
    m_test_100 = m_test[-100:].argmax(axis=1)
    r_test_100 = r_test[-100:].argmax(axis=1)
    print(pred_move)
    print(m_test_100)
    print(pred_result)
    print(pred[1])
    print(r_test_100)
    del model
