cd /d %~dp0
del /q .\kifu
start /min python bitboard.py
timeout /T 1
start /min python bitboard.py
timeout /T 1
start /min python bitboard.py
timeout /T 1
start /min python bitboard.py
timeout /T 1
start /min python bitboard.py
timeout /T 1
start /min python bitboard.py
timeout /T 1
start /min python bitboard.py
timeout /T 1
call python bitboard.py
timeout /T 10
python make_kifu_list.py kifu kifulist
python bitboard_learn.py
pause