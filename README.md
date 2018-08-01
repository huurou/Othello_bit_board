"# Othello_bit_board" 
bitboard_ucb.py:ucbで棋譜生成　棋譜数、探索数は中のKIFU_NUM, SEARCH_NUMをいじって調節して
                kifuフォルダに棋譜が生成される
make_kifu_list.py:kifuフォルダ中の棋譜を訓練データとテストデータに分け、それぞれの位置を書き込んだテキストファイルを生成
bitboard_learn.py:上で生成されたkifulistを読み込み学習する 生成モデル:model_policy_value.h5
