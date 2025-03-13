import cv2
import numpy as np


# Perspective transformation (Wider Bird’s Eye View)
def perspective_transform(frame):
   height, width = frame.shape[:2]


   # Adjusting the source points for a wider view
   src = np.float32([
       [width * 0.2, height * 0.65],  # Left lane line
       [width * 0.8, height * 0.65],  # Right lane line
       [width * 1.0, height * 0.95],  # Bottom-right (farther)
       [width * 0.0, height * 0.95]   # Bottom-left (farther)
   ])


   # Adjusting destination points to keep parallel lanes
   dst = np.float32([
       [width * 0.2, 0],     
       [width * 0.8, 0],     
       [width * 0.8, height],
       [width * 0.2, height] 
   ])


   matrix = cv2.getPerspectiveTransform(src, dst)
   inv_matrix = cv2.getPerspectiveTransform(dst, src)
   warped = cv2.warpPerspective(frame, matrix, (width, height))


   return warped, matrix, inv_matrix


# Preprocessing to extract lane lines
def preprocess_frame(frame):
   hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
   s_channel = hls[:, :, 2]  # Saturation channel for bright lane detection


   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(gray, (5, 5), 0)


   # Combine color and edge detection
   edges = cv2.Canny(blur, 50, 150)
   _, binary = cv2.threshold(s_channel, 100, 255, cv2.THRESH_BINARY)


   combined = cv2.bitwise_or(binary, edges)  # Merge both
   return combined


# Sliding window lane detection
def sliding_window_lane_detection(binary_warped):
   histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
   mid_point = np.int32(histogram.shape[0] // 2)
   left_x_base = np.argmax(histogram[:mid_point])
   right_x_base = np.argmax(histogram[mid_point:]) + mid_point


   num_windows = 10
   window_height = np.int32(binary_warped.shape[0] // num_windows)
   nonzero = binary_warped.nonzero()
   nonzero_y = np.array(nonzero[0])
   nonzero_x = np.array(nonzero[1])


   left_x_current = left_x_base
   right_x_current = right_x_base
   margin = 50
   min_pixels = 50


   left_lane_indices = []
   right_lane_indices = []


   for window in range(num_windows):
       win_y_low = binary_warped.shape[0] - (window + 1) * window_height
       win_y_high = binary_warped.shape[0] - window * window_height
       win_xleft_low = left_x_current - margin
       win_xleft_high = left_x_current + margin
       win_xright_low = right_x_current - margin
       win_xright_high = right_x_current + margin


       good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                         (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
       good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]


       left_lane_indices.append(good_left_inds)
       right_lane_indices.append(good_right_inds)


       if len(good_left_inds) > min_pixels:
           left_x_current = np.int32(np.mean(nonzero_x[good_left_inds]))
       if len(good_right_inds) > min_pixels:
           right_x_current = np.int32(np.mean(nonzero_x[good_right_inds]))


   left_lane_indices = np.concatenate(left_lane_indices)
   right_lane_indices = np.concatenate(right_lane_indices)


   left_x = nonzero_x[left_lane_indices]
   left_y = nonzero_y[left_lane_indices]
   right_x = nonzero_x[right_lane_indices]
   right_y = nonzero_y[right_lane_indices]


   left_fit = np.polyfit(left_y, left_x, 2)
   right_fit = np.polyfit(right_y, right_x, 2)


   return left_fit, right_fit


# Calculate curvature and direction
def calculate_curvature_and_direction(left_fit, right_fit, binary_warped):
   plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
   y_eval = np.max(plot_y)


   # Define conversions in x and y from pixels space to meters
   ym_per_pix = 30 / 720  # meters per pixel in y dimension
   xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


   # Fit new polynomials to x, y in world space
   left_fit_cr = np.polyfit(plot_y * ym_per_pix, left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2], 2)
   right_fit_cr = np.polyfit(plot_y * ym_per_pix, right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2], 2)


   # Calculate the radius of curvature
   left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
   right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])


   # Calculate the direction of the turn
   lane_center = (left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2] +
                  right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]) / 2
   vehicle_center = binary_warped.shape[1] / 2
   direction = "Straight" if abs(lane_center - vehicle_center) < 100 else ("Left" if lane_center < vehicle_center else "Right")


   return left_curverad, right_curverad, direction


# Draw lanes and overlay information
def draw_lanes(original_frame, binary_warped, left_fit, right_fit, inv_matrix, direction):
   height, width = binary_warped.shape
   plot_y = np.linspace(0, height-1, height)
   left_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
   right_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]


   lane_overlay = np.zeros_like(original_frame)
   left_points = np.array([np.transpose(np.vstack([left_x, plot_y]))])
   right_points = np.array([np.flipud(np.transpose(np.vstack([right_x, plot_y])))])


   lane_points = np.hstack((left_points, right_points))
   cv2.fillPoly(lane_overlay, np.int32([lane_points]), (0, 255, 0))


   lane_overlay = cv2.warpPerspective(lane_overlay, inv_matrix, (width, height))
   result = cv2.addWeighted(original_frame, 1, lane_overlay, 0.5, 0)


   # Add direction arrow
   arrow_size = 100
   if direction == "Left":
       cv2.arrowedLine(result, (width - arrow_size - 50, 50), (width - 50, 50), (0, 0, 255), 10, tipLength=0.5)
   elif direction == "Right":
       cv2.arrowedLine(result, (50, 50), (50 + arrow_size, 50), (0, 0, 255), 10, tipLength=0.5)
   else:
       cv2.arrowedLine(result, (width // 2 - arrow_size // 2, 50), (width // 2 + arrow_size // 2, 50), (0, 0, 255), 10, tipLength=0.5)


   return result


# Main function
def main():
   cap = cv2.VideoCapture("ogvld.mov")  
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break


       # Correct unpacking of three return values
       warped, matrix, inv_matrix = perspective_transform(frame)
       binary_warped = preprocess_frame(warped)


       try:
           left_fit, right_fit = sliding_window_lane_detection(binary_warped)
           left_curvature, right_curvature, direction = calculate_curvature_and_direction(left_fit, right_fit, binary_warped)
           output = draw_lanes(frame, binary_warped, left_fit, right_fit, inv_matrix, direction)
       except:
           output = frame  # If detection fails, show the original frame


       cv2.imshow('Lane Detection', output)
      
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break


   cap.release()
   cv2.destroyAllWindows()


if __name__ == "__main__":
   main()

