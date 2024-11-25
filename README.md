# Driving_Mind

AI driving mind uploaded on github:

for Team AI-FORCE

## TODO

jupyter는 그냥 로봇에서 돌리면 그만이다: 세션 자주 갈아주기 정도

pip install ultralytics keyboard 

아예 Jupyter 말고 그냥 Python을 쓰고 싶다면, 그냥 -ssh 접속 쓰기로 하자.

VS Code 안에서도 어.. 이것저것 해줄 것

### 실행을 원한다면, ssh -X username@ip_address

이렇게 해야 GUI가 내 화면에서 뜬다. 

### Model을 적용할 때 주의점

JetPack 4.6.1 / Onnx 1.11 이하, TensorRT 8.2.1, CUDA 10.2

여기서... DETR이랑 YOLO가 잘 돌아갈까? 

일단 GPU memory는 몇백 MB밖에 먹지 않는 것으로 보임.. 의외로 괜찮을지도

## Basic Shape

mind_bot.py 실행: 각각의 Mode들이 만들어짐.

매 frame마다 Mode의 set_frame_and_move 함수를 호출: frame만 넣어줌, 각 mode에 맞게 움직임

mode가 end 되면, 다음 mode를 꺼내서 씀

: mode 사이에 전해져야 하는 정보는 capsule (dict) 안에 넣어서 전달

## Mode classes

Eve (EventMode):

    이미지 촬영 및 predict, (1)
    
    결괏값 이용해서 출력, (2)
    
    적 탱크 있다면 필요한 만큼 제자리 회전까지 진행 후(3) 대포 빵, 같은 시간만큼 돌아옴(4).

    predict에 상당히 많은 frame이 밀릴 수 있으니, 이후 frame 보충해주는 시간을 줌(5). 

S2G (Stanley2Green)

    Sliding window 기반 Stanley 진행(1)

    Green 발견 시, stanley로, 속도 줄여가며 접근(2)!
    
    이후 멈추고 다음 모드로 진행.

S2C (Stanley2Cross)

    Sliding Window 기반 길 찾기 진행

    sliding window는 따라가되, 그 중심에서부터 한 쪽으로 얼마 이상 채워지면 cross로 인식

T2V(Turn2Void)

    살짝 반대로 돌았다가(1) 직후 자신의 각도 확인. 

    지정된 시간 * 돌아야 하는 각도, 측정은 길 canny -> hough로.
    
    시간에 따라 내 각도가 어떻게 변하는지 확인하고(2) : 지금은 사용하지 않음

    돌아야 하는 시간만큼 돌아감(3)

T2R(Turn2Road)

    일단 지정된 시간만큼 돌고(1), 그 뒤로 sliding window로 길 찾으면 멈추기(2)

    만약 Curve라면: cross로부터의 거리와 각도 구해서, 반경 구해서 각속도/선속도 조정 : 쭉 유지



## TODO

- 연산량이 많은 게 아니라 그냥 iteration을 python이 아니라 C에서 최대한 굴리도록 해야 함: Numpy의 함수들을 최대한 많이 활용할 것.




