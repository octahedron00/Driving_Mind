# driving_mind
AI driving mind uploaded on github:

[https://drive.google.com/file/d/1Vl1SgV8VoHQgWE8x0zETEKAes9rvTUzS/view?usp=drivesdk]


### TODO

- T2V가 지금 멀쩡한지 모르겠음. 많은 테스트 필요: 기본값 쓰기도 하니까.

- T2R 완만하게 꺾는 건 앞의 cross 값이 필요함: 아예 자기도 cross 보게 하자, 처음에는. 그러면 거리가 나옴.

- T2R 속도에 따라 크게 달라질 텐데, 간단한 공식으로 해결 가능함.

- 전체적으로 속도가 제한됨: Frame 단위 현재 2, 이걸 1로 맞출 수 있다면 2배속까지 쉽게 해결되는 문제. 

| 연산량이 많은 게 아니라 그냥 iteration을 python이 아니라 C에서 최대한 굴리도록 해야 함: Numpy의 함수들을 최대한 많이 활용할 것.



