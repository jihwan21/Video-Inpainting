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
## 4. Demo Result
- Input Video


https://github.com/jihwan21/Video-Inpainting/assets/96354328/d806852a-5a92-4a89-941e-eb62d470efd4


- Output Video


https://github.com/jihwan21/Video-Inpainting/assets/96354328/0e2e13e1-25bb-46e7-8add-01faa730aaba


---
## 5. 자료
- 발표자료  
  [Conference 발표 자료.pdf](https://drive.google.com/file/d/17PPLhjVO9o_DAe3eckQtZkmQsibEAiNz/view?usp=drive_link)  

