# Video-Inpainting
---
---
## 1. 배경 & 목적
- 주관 : 국민대학교 빅데이터 분석 학회 D&A Conference
- 프로젝트 주제 : Video Inpainting을 활용한 영상 내 특정 객체 제거
- 목적 : 영상매체에 노출되는 일반인들의 초상권 문제 존재, 영상 내 주인공을 제외한 사람들을 inpainting 기술을 활용해 지워보자

---
## 2. 담당 역할
- Team Member
- 아이디어 기획
- 선행연구 조사
---
## 3. Model Flow
![image](https://github.com/jihwan21/Video-Inpainting/assets/96354328/1ed5230e-cd10-4736-a915-dedd335e2b7d)
1 사용자가 영상과 영상 속 지우고자 하는 객체를 입력
2 YOLO-X로 Frame 별 Object Detection -> HybridSORT로 Tracking 수행
3 동시에 PIDNET으로 Frame 별로 Segmentation 수행
4 사용자가 지정한 객체의 bbox와 Segmentation 정보를 통해 masking map 생성
5 ProPainter로 Video Inpainting 진행

---
## 5. 최종 발표(팀)
- **주제 선정 배경(연구 동기)**   
산업재해 발생유형 중 부딪힘, 교통사고 약 25%  
예기치 못한 사고를 미연에 방지 , 예방하기 위해 산업 현장 내 객체 간의 충돌 예측 모델 고안  
현장 CCTV 를 통해 실시간 충돌 예측 및 경고 시스템 구축 -> 충돌로 인한 산업재해 발생률 감소 기대

- **기존 충돌 예측 연구 한계점**
  1) 하나의 기준으로 충돌 판단
  2) 객체와 객체 사이의 거리가 멀어 bbox가 겹치지 않는 경우충돌을 예측 할 수 없다는 한계 존재
  3) 작업 차량과 차량 탑승자를 충돌하는 경우로 오판단하는 경우 발생 (IoU의 값이 1에 근사하기 때문)

- **선행 연구 조사**  
객체 탐지 모델, 경로 예측 모델

- **제안 방법론 파이프라인**
  1) Multi-Object Detection & Tracking
  2) Distance Calculation & Predict Trajectory
  3) Collision Prediction & Output
 
     ![image](https://github.com/jihwan21/Real-time-collision-prediction/assets/96354328/c9a003bf-1e99-4838-9706-afbeedc04d6e)


- **실험 시나리오 설계**
---
## 6. 자료
- 발표자료  
  [선행연구 조사 발표.pdf](https://drive.google.com/file/d/1yVpEJaxesETHPn9F5NeQw57UR9uE3CxA/view?usp=drive_link)  
  [최종 발표.pdf](https://drive.google.com/file/d/11KB0dwEv6wM7rpzWCvbeLyHe5vhpmXfd/view?usp=drive_link)
