# python2
 
### django와 apache를 연동해 로컬에 미리 학습시켜 저장되어 있는 모델을 이용해 음악(abc notation) 생성, midi파일로 추출
### 현재는 mysite 경로에서 python manage.py runserver 커맨드로 apache 연동 없이 가상서버로 동작 확인 가능


<div align="center">
 <p align="center">
  [2019] 딥러닝을 이용한 BGM 작곡 설계 및 구현
 </p>
</div>
Non-Copyright Music에 대한 수요의 증가로 인해 인디 개발자들의 2차 창작물을 증가할 수 있게 비즈니스적 문제를 해결하고자 주제를 선정하였습니다.

목차
개요
작품설명
기술술명
결과음악


개요
프로젝트 이름: 딥러닝을 이용한 음악작곡(Composition Using Deep Learning)
프로젝트 기간 : 2018.10~2019.09
개발 언어 : Python
사용 라이브러리 : selenim - , ipykernal - , tensorflow - , keras - , music21 - 
모델 학습 주요코드 : mysite/music.py
장고 서버 알고리즘 코드 : polls/views.py

작품설명
초기화면 작업후 다운로드화면

기술설명
1) 사용 데이터 : abc notation
2) 학습 모델 : LSTM
3) 서비스 시스템 구성

결과 음악
1.2.3.

첨부 논문
