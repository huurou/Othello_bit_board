import os.path
import pickle

import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Activation, Concatenate, Conv2D, Dense, Dropout,
                          Flatten, Input, concatenate)
from keras.models import Model, Sequential, load_model
from tqdm import tqdm


# ビットボードをnp.array([8,8])に変換
def bitboard_convert_to_nparray(bitb):
    mask = np.uint64(0x8000000000000000)
    na = np.zeros((64), dtype='int')
    if bitb == 0:
        return na.reshape(8, 8)
    for i in range(64):
        if bitb & mask:  # maskedにビットが立ってる
            na[i] = 1
            bitb &= mask ^ 0xffffffffffffffff
            if not bitb:  # bitbがFlase->立っているビットがないのでreturn
                return na.reshape(8, 8)
        mask >>= np.uint64(1)


def readkifu(kifu_list_file):
    print(f'{kifu_list_file}を読み込んでいます')
    bl = []  # ban_list
    rl = []  # result_list
    bls = np.empty((0, 4, 8, 8))  # black, white, void, teban
    rls = np.empty((0, 2))
    cnt = 1
    with open(kifu_list_file, 'r')as f:
        for line in tqdm(f.readlines()):
            kifu = line.rstrip('\r\n')
            for i in np.load(kifu):
                i = i.astype(np.uint64)
                bb = bitboard_convert_to_nparray(i[0])  # black_board
                wb = bitboard_convert_to_nparray(i[1])  # white_board
                vb = bitboard_convert_to_nparray(i[2])  # viod_board
                teban = np.full((8, 8), i[3], dtype='int')
                ban = np.stack([bb, wb, vb, teban])
                result = np.eye(2, dtype='int')[i[5]]
                bl.append(ban)
                rl.append(result)
            cnt += 1
            if cnt % 1000 == 0:
                bls = np.append(bls, np.array(bl), axis=0)
                rls = np.append(rls, np.array(rl), axis=0)
                bl = []
                rl = []
        bls = np.append(bls, np.array(bl), axis=0)
        rls = np.append(rls, np.array(rl), axis=0)
        bl = []
        rl = []
        print(f'{kifu_list_file}を読み込みました')
    return bls, rls

if __name__ == '__main__':
    b_train, r_train = readkifu('kifulist_train.txt')
    b_test, r_test = readkifu('kifulist_test.txt')
    if os.path.isfile('model_value.h5'):
        model = load_model('model_value.h5')
        print('モデルmodel_value.h5を読み込みました')
    else:
        print('モデルはありません')
        ban_input = Input(shape=(4, 8, 8))
        x = Conv2D(192, (3, 3),
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.001),
                   activation='relu')(ban_input)
        x = Conv2D(192, (3, 3),
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.001),
                   activation='relu')(x)
        x = Conv2D(192, (3, 3),
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.001),
                   activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(256, kernel_regularizer=regularizers.l2(0.001),
                  activation='relu')(x)
        x = Dense(256, kernel_regularizer=regularizers.l2(0.001),
                  activation='relu')(x)
        x = Dense(256, kernel_regularizer=regularizers.l2(0.001),
                  activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(2, activation='softmax',
                       name='result')(x)
        model = Model(inputs=ban_input,
                      outputs=output)
    model.summary()
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=1)
    md_ch = ModelCheckpoint(filepath='model_value.h5')
    history = model.fit(b_train, r_train,
                        epochs=20, batch_size=32,
                        verbose=1, validation_data=(b_test, r_test),
                        callbacks=[early_stopping, md_ch])
    model.save('model_value.h5')
    score = model.evaluate(b_test, r_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
