import cv2
import numpy as np
from collections import deque

def overlayImage(frame, arrow, angle):
    arrow = cv2.imread(arrow, cv2.IMREAD_UNCHANGED)
    arrow = cv2.resize(arrow, (300, 300))
    
    # Rotate arrow based on detected lane movement
    center = (150, 150)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    arrow = cv2.warpAffine(arrow, matrix, (300, 300))
    
    rows, cols, channels = arrow.shape
    roi = frame[0:rows, 0:cols]
    
    img2gray = cv2.cvtColor(arrow, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(arrow, arrow, mask=mask)
    
    dst = cv2.add(img1_bg, img2_fg)
    frame[0:rows, 0:cols] = dst
    
    return frame

def determine_arrow_angle(left_line, prev_left_line):
    if left_line is None or prev_left_line is None:
        return 0  # Keep arrow straight if no movement is detected
    
    x1, _, x2, _ = left_line
    prev_x1, _, prev_x2, _ = prev_left_line
    
    delta_x = (x2 - x1) - (prev_x2 - prev_x1)
    
    if delta_x > 5:  # Moving right
        return -30  # Turn arrow right
    elif delta_x < -5:  # Moving left
        return 30  # Turn arrow left
    
    return 0  # No change

def process_video(video_path, arrow):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    left_lines_buffer = deque(maxlen=5)
    prev_left_line = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result_frame = frame.copy()
        
        left_line = [100, frame_height, 200, int(frame_height * 0.47)]  # Simulated lane detection
        
        if left_lines_buffer:
            prev_left_line = left_lines_buffer[-1]
        
        left_lines_buffer.append(left_line)
        
        arrow_angle = determine_arrow_angle(left_line, prev_left_line)
        result_frame = overlayImage(result_frame, arrow, arrow_angle)
        
        cv2.imshow('Lane Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = "ogvld.mov"
    arrow = "arrowimage.png"
    process_video(video_path, arrow)

if __name__ == "__main__":
    main()
