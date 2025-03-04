import cv2
import numpy as np
from collections import deque

def overlayImage(frame, arrow):
   arrow = cv2.imread(arrow)
   arrow = cv2.resize(arrow, (300,300))
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

def detect_white_dashed_lanes_left_column(image, roi_mask, frame_width, frame_height):
    # Apply ROI mask for bottom 53%
    masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
    
    # Create a mask for the leftmost 33% column
    left_column_mask = np.zeros_like(roi_mask)
    column_width = int(frame_width * 0.33)
    left_column_mask[:, 0:column_width] = roi_mask[:, 0:column_width]
    
    # Apply the column mask to further restrict detection area
    masked_image = cv2.bitwise_and(masked_image, masked_image, mask=left_column_mask)
    
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    
    # Filter white colors with more aggressive thresholds
    lower_white = np.array([0, 0, 180])  # Lower brightness threshold to catch more whites
    upper_white = np.array([180, 40, 255])  # Increased saturation to catch slightly off-white
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Filter yellow colors
    lower_yellow = np.array([15, 80, 150])  # Yellow in HSV
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine white and yellow masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Apply the combined mask to the original masked image
    filtered_image = cv2.bitwise_and(masked_image, masked_image, mask=combined_mask)
    
    # Convert to grayscale and apply blur
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    
    # Apply stronger contrast enhancement to make white/yellow pop
    gray = cv2.equalizeHist(gray)
    
    # Apply a smaller kernel blur to preserve edges
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Edge detection with adjusted parameters for better sensitivity
    # Using lower threshold to catch more edges and higher threshold to filter noise
    edges = cv2.Canny(blurred, 30, 200)
    
    # Dilate to enhance line segments
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)  # Increased iterations
    
    # Detect line segments using Hough transform with adjusted parameters
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30,  # Lower threshold to detect more lines
        minLineLength=15,  # Shorter minimum line length
        maxLineGap=200  # Larger max gap
    )
    
    left_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter out horizontal lines
            if abs(slope) < 0.3:
                continue
            
            # We're only interested in lines with negative slope in the left column
            if slope < 0:
                left_lines.append((x1, y1, x2, y2, slope))
    
    return left_lines

def mirror_line_to_right(left_line, frame_width):
    """
    Mirror a line from the left column to the right column with adjustments:
    - Lower y-coordinate at the bottom (closer to frame bottom)
    - Increase x-coordinate at the top (further from center)
    """
    if left_line is None:
        return None
    
    # Unpack the line coordinates
    left_x1, y1, left_x2, y2 = left_line
    
    # Calculate the distance from the left edge
    distance_x1 = left_x1
    distance_x2 = left_x2
    
    # Mirror the points to the right side with adjustments
    right_x1 = frame_width - distance_x1 - 750
    right_x2 = frame_width - distance_x2 - 15  # Move x further out (away from 0/center)
    
    # Determine which y-coordinate is closer to the bottom of the frame
    if y1 > y2:
        # y1 is closer to bottom (larger value)
        adjusted_y1 = y1 + 1200  # Lower the bottom point (larger y value)
        right_y1 = adjusted_y1
        right_y2 = y2
    else:
        # y2 is closer to bottom (larger value)
        adjusted_y2 = y2 + 1200  # Lower the bottom point (larger y value)
        right_y1 = y1
        right_y2 = adjusted_y2
    
    # Return the adjusted mirrored line
    return [right_x1, right_y1, right_x2, right_y2]

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
def draw_centerline(left_line, right_line):
    """
    Draws the centerline between the left and right lanes.
    """
    if left_line is None or right_line is None:
        return None
    
    # Unpack line coordinates
    left_x1, left_y1, left_x2, left_y2 = left_line
    right_x1, right_y1, right_x2, right_y2 = right_line
    
    # Compute midpoint between corresponding points of left and right lanes
    center_x1 = ((left_x1 + right_x1) // 2)
    center_y1 = (left_y1 + right_y1) // 2
    center_x2 = (left_x2 + right_x2) // 2
    center_y2 = (left_y2 + right_y2) // 2
    
    return [center_x1, center_y1, center_x2, center_y2]
def create_lane_polygon(line, lane_width=50):
    """
    Create a polygon (rectangle) from a line with specified width
    """
    if line is None:
        return None
    
    x1, y1, x2, y2 = line
    
    # Calculate perpendicular vector to the line
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx*dx + dy*dy)
    
    # Normalize and rotate 90 degrees to get perpendicular vector
    perpendicular_x = -dy / length
    perpendicular_y = dx / length
    
    # Calculate the four corners of the rectangle
    half_width = lane_width / 2
    points = np.array([
        [int(x1 + perpendicular_x * half_width), int(y1 + perpendicular_y * half_width)],
        [int(x2 + perpendicular_x * half_width), int(y2 + perpendicular_y * half_width)],
        [int(x2 - perpendicular_x * half_width), int(y2 - perpendicular_y * half_width)],
        [int(x1 - perpendicular_x * half_width), int(y1 - perpendicular_y * half_width)]
    ], dtype=np.int32)
    
    return points

def draw_transparent_polygon(image, polygon, color, alpha=0.4):
    """
    Draw a semi-transparent polygon on the image
    """
    if polygon is None:
        return image
    
    # Create a copy of the image
    overlay = image.copy()
    
    # Draw the filled polygon on the overlay
    cv2.fillPoly(overlay, [polygon], color)
    
    # Blend the overlay with the original image using alpha blending
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return result

def process_video(video_path, arrow):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define ROI for bottom 53% of the frame
    bottom_y = frame_height
    top_y = int(frame_height * 0.47)  # 100% - 53% = 47%

    # Define the full width ROI
    bottom_left_x = 0
    bottom_right_x = frame_width
    top_left_x = 0
    top_right_x = frame_width

    roi_points = np.array([
        [(bottom_left_x, bottom_y),
         (top_left_x, top_y),
         (top_right_x, top_y),
         (bottom_right_x, bottom_y)]
    ], dtype=np.int32)

    # Create the ROI mask
    roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, roi_points, 255)

    # Buffers for smoothing
    left_lines_buffer = deque(maxlen=5)
    
    # Define default lane positions (when no lanes are detected)
    left_column_width = int(frame_width * 0.33)
    default_left_bottom_x = int(left_column_width * 0.7)  # 70% of the left column width
    default_left_top_x = int(left_column_width * 0.3)     # 30% of the left column width
    default_left_line = [default_left_bottom_x, bottom_y, default_left_top_x, top_y]
    
    # Default right lane is the mirror of the default left lane
    default_right_line = mirror_line_to_right(default_left_line, frame_width)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = frame.copy()

        # Detect lines in left column
        left_lines = detect_white_dashed_lanes_left_column(frame, roi_mask, frame_width, frame_height)

        # Process left lane
        avg_left = None
        if left_lines:
            avg_left = average_lines(left_lines, left_lines_buffer, frame_height, top_y)
            if avg_left:
                left_lines_buffer.append(avg_left)
        
        # If no valid left lane is detected, use the default lane
        if not avg_left and (not left_lines_buffer or len(left_lines_buffer) == 0):
            avg_left = default_left_line
        elif not avg_left and left_lines_buffer:
            # Use the most recent valid lane if available
            avg_left = left_lines_buffer[-1]
        
        # Mirror left lane to right lane
        avg_right = mirror_line_to_right(avg_left, frame_width) if avg_left else None
        

        center_line = draw_centerline(avg_left, avg_right)
        center_line_polygon = create_lane_polygon(center_line, lane_width=75)
        result_frame = draw_transparent_polygon(result_frame, center_line_polygon, (0, 0, 0), alpha=.75)
                
        # Create transparent rectangles for the lanes
        if avg_left:
            # Create lane polygons
            left_lane_polygon = create_lane_polygon(avg_left, lane_width=100)
            # Draw left lane as transparent rectangle (green)
            result_frame = draw_transparent_polygon(result_frame, left_lane_polygon, (255, 255, 255), alpha=.75)
        
        if avg_right:
            # Create lane polygons
            right_lane_polygon = create_lane_polygon(avg_right, lane_width=100)
            # Draw right lane as transparent rectangle (green)
            result_frame = draw_transparent_polygon(result_frame, right_lane_polygon, (255, 255, 255), alpha=.75)
        
        # Visualize the lane detection regions (optional)
        # Left column (33%)
        left_column_width = int(frame_width * 0.33)
        #cv2.rectangle(result_frame, (0, top_y), (left_column_width, bottom_y), (255, 0, 0), 2)
        # Right column (33%)
        right_column_start = int(frame_width * 0.67)
        #cv2.rectangle(result_frame, (right_column_start, top_y), (frame_width, bottom_y), (255, 0, 0), 2)
        
        # Apply the arrow overlay
        result_frame = overlayImage(result_frame, arrow)
        
        # Display the result
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
