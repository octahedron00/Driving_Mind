import math
import time
# tiki import는 안 하기로 했습니다... 너무 복잡해져서 



# 하늘을 달리는 우리 꿈을 보아라
# 반음 기준으로 멜로디를 미리 만들어두기
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
            tiki.stop_buzzer()
            # 수학적으로 음계를 정의하는 방법
            freq = FREQ_C * math.exp2(note/12)
            tiki.play_buzzer(int(freq))

        time.sleep(0.16)

    tiki.stop_buzzer()

