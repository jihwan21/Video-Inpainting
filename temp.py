import cv2

def resize_video_resolution(video_path, output_path, scale_percent):
    """
    비디오의 해상도를 지정된 백분율로 줄입니다.
    
    :param video_path: 원본 비디오 파일 경로
    :param output_path: 해상도가 조정된 비디오를 저장할 경로
    :param scale_percent: 해상도를 줄일 백분율 (예: 50은 50%로 줄임을 의미)
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_percent / 100)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_percent / 100)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)

    cap.release()
    out.release()

# 비디오 파일 경로, 출력 파일 경로 및 해상도 조정 백분율 설정
video_path = "./videos/street_2.mp4"
output_path = "./videos/resized_street_2.mp4"
scale_percent = 50  # 예: 원본 해상도의 50%

# 함수 호출
resize_video_resolution(video_path, output_path, scale_percent)

