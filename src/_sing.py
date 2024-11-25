import math
from src.fake_tiki import TikiMini
import time


SONG_NOTE = [
12,  0,  0,  0,  0,  0, 11,  0,  0,  9,  0,  0,  7,  0,  9,  7,  0,  4,  7,  0,  0,  0,  0,  0,
12,  0,  0,  0,  0, 16, 19,  0, 16, 12,  0, 16, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
16,  0,  0,  0,  0,  0, 12,  0,  0,  7,  0,  0,  9,  0,  0,  7,  0,  4,  7,  0,  0,  0,  0,  0,
12,  0,  0,  0,  0, 16, 19,  0, 16, 14,  0, 16, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
14,  0, 14, 14,  0,  0,  0,  0,  0,  7,  0,  0, 16,  0, 16, 16,  0,  0,  0,  0,  0, 12,  0,  0,
16,  0,  0, 17,  0,  0, 19,  0,  0, 21,  0, 19, 19,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
21,  0,  0,  0,  0,  0, 19,  0,  0, 16,  0,  0, 14,  0, 12, 11,  0,  0,  9,  0,  0,  0,  0,  0,
 7,  0,  0,  0,  0,  9, 11,  0, 12, 14,  0, 16, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
]

FREQ_C = 522

def sing(tiki):
    
    for note in SONG_NOTE:
        if note > 0:
            freq = FREQ_C * math.exp2(note/12)
            tiki.play_buzzer(int(freq))

        time.sleep(0.16)

    tiki.stop_buzzer()


if __name__ == "__main__":

    sing(TikiMini())
