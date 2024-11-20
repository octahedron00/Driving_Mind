# driving_mind

AI driving mind uploaded on github:

for Team AI-FORCE

## Mode classes

Eve (EventMode):

    이미지 촬영 및 predict, (1)
    
    결괏값 이용해서 출력, (2)
    
    적 탱크 있다면 필요한 만큼 제자리 회전까지 진행 후(3) 대포 빵, 돌아옴(4).

    predict에 상당히 많은 frame이 밀릴 수 있으니, 이후 frame 보충해주는 시간을 줌(5). 

S2G (Stanley2Green)

    Sliding window 기반 Stanley 진행
    
    이후 멈추고 다음 모드로 진행.

S2C (Stanley2Cross)

    Sliding Window 기반 길 찾기 진행

    sliding window는 따라가되, 그 중심에서부터 한 쪽으로 얼마 이상 채워지면 cross로 인식

T2V(Turn2Void)

    살짝 반대로 돌았다가 돌아가며 확인함(1). 

    지정된 시간 * 돌아야 하는 각도, 측정(2)은 길 canny -> hough로.

T2R(Turn2Road)

    일단 지정된 시간만큼 돌고(1), 그 뒤로 sliding window로 길 찾으면 멈추기(2)

    만약 Curve라면: cross로부터의 거리와 각도 구해서, 반경 구해서 각속도/선속도 조정 : 쭉 유지



## TODO

| 연산량이 많은 게 아니라 그냥 iteration을 python이 아니라 C에서 최대한 굴리도록 해야 함: Numpy의 함수들을 최대한 많이 활용할 것.

- S2G에서, 속도 자체를 PID 등 제어로 green 접근 시 조정해줘야 하나? 지금보다 가시거리가 길어지면 고려해야 할 수도 있음

- Cam 왜곡이 상당히 심해보인다. 이거를 거기서 chessboard 같은 걸로 보정할 수 있나? 태블릿 이미지 띄워야 하나?


