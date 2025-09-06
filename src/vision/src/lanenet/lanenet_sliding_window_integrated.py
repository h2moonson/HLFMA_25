#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
from model2 import Lanenet

class LanenetSlidingWindowIntegrated:
    def __init__(self):
        # --- 슬라이딩 윈도우 파라미터 ---
        self.nwindows = 12
        self.win_width = 100 // 2
        self.win_height = 0  # ROI 높이에 따라 동적으로 계산됨
        self.threshold = 0   # 윈도우 내 최소 픽셀 수 (동적 계산)
        
        # --- Lanenet 모델 설정 ---
        self.MODEL_PATH = '/Users/seongjinjeong/ieve_2025/src/vision/src/lanenet/lanenet_.model'
        self.VIDEO_PATH = '/Users/seongjinjeong/ieve_2025/src/vision/cw.mp4'
        self.DEVICE = 'cpu'
        
        # 모델 로드
        print("[INFO] LaneNet 모델 로드 중...")
        self.model = Lanenet(2, 4)
        self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=torch.device(self.DEVICE)))
        self.model.to(self.DEVICE)
        self.model.eval()
        print("[INFO] 모델 로드 완료.")

    def lanenet_inference(self, image):
        """Lanenet 모델로 차선 검출 수행"""
        org_shape = image.shape
        resized = cv2.resize(image, (512, 256))
        normalized = resized / 127.5 - 1.0
        input_tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            binary_logits, _ = self.model(input_tensor)
            binary_logits = binary_logits.cpu()
            binary_img = torch.argmax(binary_logits, dim=1).squeeze().numpy()
            binary_img[0:65, :] = 0  # 상단 노이즈 제거
            binary_img = (binary_img > 0).astype(np.uint8) * 255

        # 흑백 → BGR 변환
        result = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        result = cv2.resize(result, (org_shape[1], org_shape[0]))
        return result

    def region_of_interest(self, img, vertices=None):
        """ROI 마스킹"""
        mask = np.zeros_like(img)
        if vertices is None:
            # 확장된 ROI 설정: 전방으로 더 넓은 사다리꼴
            h, w = img.shape[:2]
            vertices = np.array([[
                (0, h),                           # 왼쪽 아래
                (w * 0.1, int(h * 0.4)),        # 왼쪽 위 (전방 확장)
                (w * 0.9, int(h * 0.4)),        # 오른쪽 위 (전방 확장)
                (w, h)                           # 오른쪽 아래
            ]], dtype=np.int32)

        cv2.fillPoly(mask, vertices, 255)
        masked = cv2.bitwise_and(img, mask)
        return masked

    def sliding_window(self, img):
        """슬라이딩 윈도우 알고리즘"""
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        peak_threshold = 50 
        if histogram[left_base] < peak_threshold:
            left_base = -1 
        if histogram[right_base] < peak_threshold:
            right_base = -1

        left_current = left_base
        right_current = right_base

        left_valid_cnt, right_valid_cnt = 0, 0
        
        # 각 윈도우의 중심점과 모든 픽셀 인덱스를 저장할 리스트 생성
        left_centers, right_centers = [], []
        all_left_inds, all_right_inds = [], []

        # 슬라이딩 윈도우 결과를 그릴 이미지 복사
        crop_visualization = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for window in range(self.nwindows):
            if left_current != -1:
                win_y_low = img.shape[0] - (window + 1) * self.win_height
                win_y_high = img.shape[0] - window * self.win_height
                win_left_low  = left_current - self.win_width
                win_left_high = left_current + self.win_width
                
                cv2.rectangle(crop_visualization, (win_left_low, win_y_low), (win_left_high, win_y_high), (0, 255, 255), 2)

                good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                                  (nonzero_x >= win_left_low) & (nonzero_x < win_left_high)).nonzero()[0]
                all_left_inds.append(good_left_inds)

                if len(good_left_inds) > self.threshold:
                    left_current = int(np.mean(nonzero_x[good_left_inds]))
                    left_centers.append(left_current)
                    left_valid_cnt += 1

            if right_current != -1:
                win_y_low = img.shape[0] - (window + 1) * self.win_height
                win_y_high = img.shape[0] - window * self.win_height
                win_right_low = right_current - self.win_width
                win_right_high = right_current + self.win_width

                cv2.rectangle(crop_visualization, (win_right_low, win_y_low), (win_right_high, win_y_high), (255, 255, 0), 2)
                
                good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                                   (nonzero_x >= win_right_low) & (nonzero_x < win_right_high)).nonzero()[0]
                all_right_inds.append(good_right_inds)

                if len(good_right_inds) > self.threshold:
                    right_current = int(np.mean(nonzero_x[good_right_inds]))
                    right_centers.append(right_current)
                    right_valid_cnt += 1

        # 시각화를 위해 전체 픽셀 좌표 계산
        if len(all_left_inds) > 0:
            left_lane_inds = np.concatenate(all_left_inds)
            left_x, left_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
            if len(left_x) > 0:
                crop_visualization[left_y, left_x] = [255, 0, 0]

        if len(all_right_inds) > 0:
            right_lane_inds = np.concatenate(all_right_inds)
            right_x, right_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]
            if len(right_x) > 0:
                crop_visualization[right_y, right_x] = [0, 0, 255]

        return left_valid_cnt, right_valid_cnt, left_centers, right_centers, crop_visualization

    def process_frame(self, frame):
        """한 프레임을 처리하는 메인 함수"""
        img_h, img_w = frame.shape[:2]

        # 1. Lanenet 추론
        binary_result = self.lanenet_inference(frame)
        
        # 2. ROI 마스킹
        binary_gray = cv2.cvtColor(binary_result, cv2.COLOR_BGR2GRAY)
        roi_result = self.region_of_interest(binary_gray)
        
        # 3. 왜곡보정 (IPM - Inverse Perspective Mapping)
        src = np.array([
            [img_w * 0.25, img_h * 0.7],
            [img_w * 0.75, img_h * 0.7],
            [img_w * 1.0, img_h * 0.95],
            [img_w * 0.0, img_h * 0.95]
        ], dtype=np.float32)

        ipm_w, ipm_h = 600, 500
        dst = np.array([[0, 0], [ipm_w, 0], [ipm_w, ipm_h], [0, ipm_h]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        ipm = cv2.warpPerspective(roi_result, M, (ipm_w, ipm_h), flags=cv2.INTER_LINEAR)

        # 디버그용 소스 영역 시각화
        debug_img = cv2.polylines(frame.copy(), [src.astype(int)], True, (0, 255, 0), 2)

        # 4. 슬라이딩 윈도우 적용
        crop = ipm
        self.win_height = crop.shape[0] // self.nwindows
        self.threshold = int(self.win_width * 2 * self.win_height * 0.05)

        # 슬라이딩 윈도우 실행
        left_valid_cnt, right_valid_cnt, left_centers, right_centers, sliding_window_result = self.sliding_window(crop)

        # 5. 차선 위치 계산
        offset = 210  # 튜닝 필요
        valid_threshold = 8
        target_window_idx = 0
        lane_valid = 0

        # 양쪽 차선 모두 인식
        if left_valid_cnt > valid_threshold and right_valid_cnt > valid_threshold and len(left_centers) > target_window_idx and len(right_centers) > target_window_idx:
            left_x = left_centers[target_window_idx]
            right_x = right_centers[target_window_idx]
            mid = (left_x + right_x) // 2
            lane_valid = mid * 2 + 1
            cv2.circle(sliding_window_result, (mid, ipm_h - 10), 10, (0, 255, 0), -1)

        # 왼쪽 차선만 인식
        elif left_valid_cnt > valid_threshold and len(left_centers) > target_window_idx:
            left_x = left_centers[target_window_idx]
            mid = left_x + offset
            lane_valid = mid * 2 + 1
            cv2.circle(sliding_window_result, (mid, ipm_h - 10), 10, (255, 0, 0), -1)

        # 오른쪽 차선만 인식
        elif right_valid_cnt > valid_threshold and len(right_centers) > target_window_idx:
            right_x = right_centers[target_window_idx]
            mid = right_x - offset
            lane_valid = mid * 2 + 1
            cv2.circle(sliding_window_result, (mid, ipm_h - 10), 10, (0, 0, 255), -1)

        else:
            # 차선 미인식
            lane_valid = 0

        return {
            'original': frame,
            'lanenet_result': binary_result,
            'roi_result': roi_result,
            'ipm_result': ipm,
            'sliding_window_result': sliding_window_result,
            'debug_img': debug_img,
            'lane_valid': lane_valid,
            'left_valid_cnt': left_valid_cnt,
            'right_valid_cnt': right_valid_cnt
        }

    def run(self):
        """메인 실행 함수"""
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        if not cap.isOpened():
            print(f"[ERROR] 영상 열기 실패: {self.VIDEO_PATH}")
            return

        # FPS 얻기
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_sec = 30
        start_frame = int(fps * start_sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[INFO] {start_sec}초 지점부터 영상 시작 (프레임 번호: {start_frame})")
        print(f"[INFO] 원본 FPS: {fps}, 2배 빠른 재생을 위해 대기시간: {int(1000/fps/2)}ms")

        print("[INFO] 영상 추론 시작... ESC 눌러 종료")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 처리
            results = self.process_frame(frame)

            # 결과 시각화
            cv2.imshow("Original Frame", results['original'])
            cv2.imshow("LaneNet Result", results['lanenet_result'])
            cv2.imshow("ROI Result", results['roi_result'])
            cv2.imshow("IPM Result", results['ipm_result'])
            cv2.imshow("Sliding Window", results['sliding_window_result'])
            cv2.imshow("Debug Source Area", results['debug_img'])

            # 정보 출력
            print(f'Lane Valid: {results["lane_valid"]} | Left Cnt: {results["left_valid_cnt"]} | Right Cnt: {results["right_valid_cnt"]}')

            # 2배 빠른 재생을 위한 대기시간 계산
            wait_time = max(1, int(1000/fps/2))  # 최소 1ms 보장
            if cv2.waitKey(wait_time) & 0xFF == 27:  # ESC 누르면 종료
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 종료됨.")

def main():
    processor = LanenetSlidingWindowIntegrated()
    processor.run()

if __name__ == "__main__":
    main() 