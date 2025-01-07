import cv2
import numpy as np

previous_lines = []
previous_mid_x = None
alpha = 0.9

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def smooth_value(new_value, previous_value, alpha=0.9):
    if previous_value is None:
        return new_value
    return int(alpha * previous_value + (1 - alpha) * new_value)

def filter_lines(lines, angle_threshold=20):
    global previous_lines
    
    if lines is None:
        if previous_lines:
            return previous_lines
        return []
        
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    
        if 90 - angle_threshold <= angle <= 90 + angle_threshold:
            if y2 > y1:  
                x1, y1, x2, y2 = x2, y2, x1, y1
            filtered_lines.append([x1, y1, x2, y2])
            
        
        if angle <= angle_threshold or angle >= 180 - angle_threshold:
            if x2 < x1:  
                x1, y1, x2, y2 = x2, y2, x1, y1
            filtered_lines.append([x1, y1, x2, y2])
    
    if filtered_lines:
        previous_lines = filtered_lines
    elif previous_lines:
        return previous_lines
            
    return filtered_lines

def get_two_main_lines(lines, frame_width):
    if len(lines) < 2:
        return []
    
   
    lines.sort(key=lambda line: line[0])
    
    left_lines = []
    right_lines = []
    mid_x = frame_width // 2
    
    for line in lines:
        x1 = line[0]
        if x1 < mid_x:
            left_lines.append(line)
        else:
            right_lines.append(line)
    
    if left_lines and right_lines:
        left_line = max(left_lines, key=lambda line: line[0])
        right_line = min(right_lines, key=lambda line: line[0])
        return [left_line, right_line]
    return []

def draw_lines_and_center(frame, lines):
    global previous_mid_x
    line_image = np.zeros_like(frame)
    

    for x1, y1, x2, y2 in lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    
    vertical_lines = [line for line in lines if abs(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0]))) > 70]
    horizontal_lines = [line for line in lines if abs(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0]))) < 20]
    
    
    main_vertical_lines = get_two_main_lines(vertical_lines, frame.shape[1])
    if len(main_vertical_lines) == 2:
        left_x = main_vertical_lines[0][0]
        right_x = main_vertical_lines[1][0]
        mid_x = int((left_x + right_x) / 2)
        mid_x = smooth_value(mid_x, previous_mid_x)
        previous_mid_x = mid_x
        
        height = frame.shape[0]
        cv2.line(line_image, (mid_x, 0), (mid_x, height), (0, 0, 255), 3)
    
    
    if len(horizontal_lines) >= 2:
        horizontal_lines.sort(key=lambda line: line[1])
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
        
        mid_y = int((top_line[1] + bottom_line[1]) / 2)
        width = frame.shape[1]
        cv2.line(line_image, (0, mid_y), (width, mid_y), (0, 0, 255), 3)
    
    return line_image

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)
        
        lines = cv2.HoughLinesP(edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=80,
            maxLineGap=10
        )

        filtered_lines = filter_lines(lines)
        if filtered_lines:
            line_image = draw_lines_and_center(frame, filtered_lines)
            frame = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

        cv2.imshow('Line Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
