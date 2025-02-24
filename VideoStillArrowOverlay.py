import cv2
import numpy as np
from collections import deque


def find_turns():
    pass
def overlayImage(frame, arrow):
    arrow = cv2.imread(arrow)
    arrow = cv2.resize(arrow, (300,300))
    rows, cols, channels = arrow.shape
    roi = frame[0:rows, 0:cols]

    overlayImage = cv2.resize(arrow, (100, 100))
    img2gray = cv2.cvtColor(arrow, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(arrow, arrow, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    frame[0:rows, 0:cols] = dst

    return frame

def find_and_fit_lane_lines(image, roi_mask, frame_height):
    masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
    
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            if abs(slope) < 0.3:
                continue

            if slope < 0:
                left_lines.append((x1, y1, x2, y2, slope))
            else:
                right_lines.append((x1, y1, x2, y2, slope))

    return left_lines, right_lines

def average_lines(lines, prev_lines, frame_height, top_y):
    if not lines:
        return None

    x1s, y1s, x2s, y2s, slopes = zip(*lines)
    avg_slope = np.mean(slopes)

    x1, y1, x2, y2 = np.mean(x1s), np.mean(y1s), np.mean(x2s), np.mean(y2s)

    if prev_lines:
        prev = np.mean(prev_lines, axis=0)
        x1 = 0.7 * prev[0] + 0.3 * x1
        y1 = 0.7 * prev[1] + 0.3 * y1
        x2 = 0.7 * prev[2] + 0.3 * x2
        y2 = 0.7 * prev[3] + 0.3 * y2
        avg_slope = (y2 - y1) / (x2 - x1 + 0.0001)

    if avg_slope != 0:
        bottom_x = int(x1 + (frame_height - y1) / avg_slope)
        top_x = int(x1 + (top_y - y1) / avg_slope)

        return [bottom_x, frame_height, top_x, top_y]

    return None

def process_video(video_path, arrow):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define trapezoid ROI points
    trap_bottom_width = 1
    trap_top_width = 1
    trap_height = 0.53

    bottom_y = frame_height
    top_y = int(frame_height * (1 - trap_height))

    bottom_left_x = int(frame_width * ((1 - trap_bottom_width) / 2))
    bottom_right_x = int(frame_width * ((1 + trap_bottom_width) / 2))
    top_left_x = int(frame_width * ((1 - trap_top_width) / 2))
    top_right_x = int(frame_width * ((1 + trap_top_width) / 2))

    roi_points = np.array([
        [(bottom_left_x, bottom_y),
         (top_left_x, top_y),
         (top_right_x, top_y),
         (bottom_right_x, bottom_y)]
    ], dtype=np.int32)

    # Buffers for smoothing
    left_lines_buffer = deque(maxlen=5)
    right_lines_buffer = deque(maxlen=5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = frame.copy()

        # Create ROI mask
        mask = np.zeros_like(frame[:, :, 0])
        cv2.fillPoly(mask, roi_points, 255)

        # Find lines
        left_lines, right_lines = find_and_fit_lane_lines(frame, mask, frame_height)

        # Average and smooth lines
        if left_lines:
            avg_left = average_lines(left_lines, left_lines_buffer, frame_height, top_y)
            if avg_left:
                left_lines_buffer.append(avg_left)
                cv2.line(result_frame, (avg_left[0], avg_left[1]), (avg_left[2], avg_left[3]), (0, 255, 0), 10)

        if right_lines:
            avg_right = average_lines(right_lines, right_lines_buffer, frame_height, top_y)
            if avg_right:
                right_lines_buffer.append(avg_right)
                cv2.line(result_frame, (avg_right[0], avg_right[1]), (avg_right[2], avg_right[3]), (0, 255, 0), 10)
        result_frame = overlayImage(result_frame, arrow)
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
