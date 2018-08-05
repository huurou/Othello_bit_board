import math
import random
from datetime import datetime

import numpy as np
from tqdm import tqdm

from gmpy2 import popcount

KIFU_NUM = 10000

# kifu:[black_board, white_board, void_board, teban, move_value, result]


# デバッグ用
def ban(b):
    t = format(b, '064b')
    print(t[0:8])
    print(t[8:16])
    print(t[16:24])
    print(t[24:32])
    print(t[32:40])
    print(t[40:48])
    print(t[48:56])
    print(t[56:64])
    print()


# boardクラス 黒盤と白盤の情報を持つ
class Board:
    def __init__(self, black_board, white_board):
        self.bb = black_board
        self.wb = white_board


# nodeクラス 手番をプレイヤー盤及び敵盤に黒盤があるか白盤があるかで示す
class Node:
    def __init__(self, player_board, oponent_board):
        self.pb = player_board  # 手番側の盤
        self.ob = oponent_board  # 相手側の盤
        self.cn = []  # 子ノード格納リスト


# 盤についてのBanクラス
class Ban:
    # 初期配置が入ったboardクラスを返す
    def init_ban(self):
        bb = 0x0000000810000000
        wb = 0x0000001008000000
        return Board(bb, wb)


# 盤面操作などのゲーム動作についてのplayクラス
class Play:
    # 手番側の盤、相手側の盤を受け取り
    # 手番側の合法手の位置のビットだけ立てた盤を返す
    def legal_board(self, pb, ob):
        hb = ob & 0x7e7e7e7e7e7e7e7e  # 左右方向の番兵
        vb = ob & 0x00ffffffffffff00  # 上下方向の番兵
        ab = ob & 0x007e7e7e7e7e7e00  # 全方向の番兵
        bl = ~(pb | ob)  # blank 石がない場所にビットを立てた盤

        # 各8方向にそれぞれ6回シフトし
        # 隣に相手の石がある石についてビットを立てる
        # 左方向
        t = hb & (pb << 1)
        for _ in range(5):
            t |= hb & (t << 1)
        lb = bl & (t << 1)  # lb:legal_board
        # 右方向
        t = hb & (pb >> 1)
        for _ in range(5):
            t |= hb & (t >> 1)
        lb |= bl & (t >> 1)
        # 上方向
        t = vb & (pb << 8)
        for _ in range(5):
            t |= vb & (t << 8)
        lb |= bl & (t << 8)
        # 下方向
        t = vb & (pb >> 8)
        for _ in range(5):
            t |= vb & (t >> 8)
        lb |= bl & (t >> 8)
        # 左上方向
        t = ab & (pb << 9)
        for _ in range(5):
            t |= ab & (t << 9)
        lb |= bl & (t << 9)
        # 右上方向
        t = ab & (pb << 7)
        for _ in range(5):
            t |= ab & (t << 7)
        lb |= bl & (t << 7)
        # 左下方向
        t = ab & (pb >> 7)
        for _ in range(5):
            t |= ab & (t >> 7)
        lb |= bl & (t >> 7)
        # 右下方向
        t = ab & (pb >> 9)
        for _ in range(5):
            t |= ab & (t >> 9)
        lb |= bl & (t >> 9)

        return lb

    # legal_boardをmove_positionに変換し、リストに格納
    def legal_list(self, pb, ob):
        lb = self.legal_board(pb, ob)
        ll = []  # legal_list
        while lb:
            t = lb & -lb
            ll.append(t)
            lb &= ~t
        return ll

    # move_positionがlegalかどうか判定
    def is_legal(self, pb, ob, mp):
        lb = self.legal_board(pb, ob)
        if (lb & mp) == mp:
            return True
        return False

    # move_positionとdirectionを受け取り
    # reverse_positionを返す
    def transfer(self, mp, dir):
        dd = {0: (mp << 8) & 0xffffffffffffff00,  # 上方向
              1: (mp << 7) & 0x7f7f7f7f7f7f7f00,  # 右上方向
              2: (mp >> 1) & 0x7f7f7f7f7f7f7f7f,  # 右方向
              3: (mp >> 9) & 0x007f7f7f7f7f7f7f,  # 右下方向
              4: (mp >> 8) & 0x00ffffffffffffff,  # 下方向
              5: (mp >> 7) & 0x00fefefefefefefe,  # 左下方向
              6: (mp << 1) & 0xfefefefefefefefe,  # 左方向
              7: (mp << 9) & 0xfefefefefefefe00}  # 左上方向
        return dd[dir]

    # player_board,opponent,move_positionを受け取り
    # 着手後のplayer_board,opponent_boardを返す
    def put_turn(self, pb, ob, mp):
        rev = 0
        for dir in range(8):
            rev_ = 0
            mask = self.transfer(mp, dir)
            while (mask != 0) and ((mask & ob) != 0):
                rev_ |= mask
                mask = self.transfer(mask, dir)
            if (mask & pb) != 0:
                rev |= rev_
        pb ^= mp | rev
        ob ^= rev
        return pb, ob

    # pb,obをcolorをもとにboard.bb,board.wbにセットする
    def set_board(self, pb, ob, color):
        if color == 0:  # 黒番
            board = Board(pb, ob)
        elif color == 1:  # 白番
            board = Board(ob, pb)
        return board


class Move:
    # legal_listからランダムなmove_positionを返す
    def rand_move(self, pb, ob):
        ll = Play().legal_list(pb, ob)
        mp = random.choice(ll)
        return mp


class Othello:
    def cvc_kifu(self, count):
        ban = Ban().init_ban()
        color = 0
        pb = ban.bb
        ob = ban.wb
        kifulist_ = []
        while True:
            if not Play().legal_board(pb, ob):
                mv = 64
                vb = (ban.bb | ban.wb) ^ 0xffffffffffffffff
                kifu = np.array([ban.bb, ban.wb, vb, color, mv],
                                dtype='uint64')
                kifulist_.append(kifu)
                color ^= 1
                pb, ob = ob, pb
                if not Play().legal_board(pb, ob):
                    mv = 64
                    vb = (ban.bb | ban.wb) ^ 0xffffffffffffffff
                    kifu = np.array([ban.bb, ban.wb, vb, color, mv],
                                    dtype='uint64')
                    kifulist_.append(kifu)
                    break
            mp = Move().rand_move(pb, ob)
            # move_positionをmove_valueに変換
            mv = 63 - popcount(mp - 1)
            vb = (ban.bb | ban.wb) ^ 0xffffffffffffffff  # void_board
            kifu = np.array([ban.bb, ban.wb, vb, color, mv],
                            dtype='uint64')
            kifulist_.append(kifu)
            # player_board, opponent_board更新
            pb, ob = Play().put_turn(pb, ob, mp)
            # board更新
            ban = Play().set_board(pb, ob, color)
            # 手番とplayer_board, opponent_board入れ替え
            color ^= 1
            pb, ob = ob, pb
        kifulist = np.array(kifulist_, dtype='uint64')
        bs = popcount(ban.bb)
        ws = popcount(ban.wb)
        row_num = kifulist.shape[0]
        if bs >= ws:
            r = np.zeros((row_num, 1))
            kifulist = np.append(kifulist, r, axis=1)
        elif bs < ws:
            r = np.ones((row_num, 1))
            kifulist = np.append(kifulist, r, axis=1)
        np.save('./kifu/{0}-{1}'.format(
            datetime.now().strftime(
                '%y_%m_%d_%H_%M_%S_%f'
                ), count), kifulist)

if __name__ == '__main__':
    for count in tqdm(range(1, KIFU_NUM + 1)):
        Othello().cvc_kifu(count)
