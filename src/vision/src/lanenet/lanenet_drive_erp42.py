#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
from model2 import Lanenet  # 사용자 정의 모델
import os
# from driving_es import region_of_interest

def region_of_interest(img, vertices=None):
    mask = np.zeros_like(img)
    if vertices is None:
        # 기본 ROI 설정: 하단 삼각형
        h, w = img.shape[:2]
        vertices = np.array([[
            (0, h),
            (w // 2, int(h * 0.6)),
            (w, h)
        ]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def warp_perspective(img):
    h, w = img.shape[:2]
    print(f"[DEBUG] warp_perspective input shape: {img.shape}")

    src_points = np.float32([
        [200, h],       # 왼쪽 아래
        [550, 450],     # 왼쪽 위
        [730, 450],     # 오른쪽 위
        [1080, h]       # 오른쪽 아래
    ])
    print(f"[DEBUG] src_points: {src_points}")

    dst_points = np.float32([
        [w//4, h],      # 왼쪽 아래
        [w//4, 0],      # 왼쪽 위
        [w//4*3, 0],    # 오른쪽 위
        [w//4*3, h]     # 오른쪽 아래
    ])
    print(f"[DEBUG] dst_points: {dst_points}")

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, (w, h))
    print(f"[DEBUG] warp_perspective output shape: {warped.shape}")
    print(f"Warped min/max: {warped.min()}/{warped.max()}")
    return warped



# 모델 경로와 비디오 경로 설정
MODEL_PATH = '/Users/seongjinjeong/iscc_2024/src/camera/lanenet/src/lanenet_.model'
VIDEO_PATH = '/Users/seongjinjeong/iscc_2024/src/camera/lanenet/cw.mp4'

# 디바이스는 CPU로 고정
DEVICE = 'cpu'

def inference(model, device, image):
    org_shape = image.shape
    resized = cv2.resize(image, (512, 256))
    normalized = resized / 127.5 - 1.0
    input_tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        binary_logits, _ = model(input_tensor)
        binary_logits = binary_logits.cpu()
        binary_img = torch.argmax(binary_logits, dim=1).squeeze().numpy()
        binary_img[0:65, :] = 0  # 상단 노이즈 제거
        binary_img = (binary_img > 0).astype(np.uint8) * 255

    # 흑백 → BGR 변환
    result = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    result = cv2.resize(result, (org_shape[1], org_shape[0]))
    return result

def main():
    print("[INFO] LaneNet 모델 로드 중...")
    model = Lanenet(2, 4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval()
    print("[INFO] 모델 로드 완료.")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {VIDEO_PATH}")
        return

    # FPS 얻기
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_sec = 30
    start_frame = int(fps * start_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"[INFO] {start_sec}초 지점부터 영상 시작 (프레임 번호: {start_frame})")

    print("[INFO] 영상 추론 시작... ESC 눌러 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lanenet 추론
        binary = inference(model, DEVICE, frame)

        # ROI 마스킹
        binary_gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        roi = region_of_interest(binary_gray)

        # 결과 출력
        cv2.imshow("Original Frame", frame)
        cv2.imshow("LaneNet Binary ROI", roi)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 종료됨.")
    
if __name__ == "__main__":
    main()
