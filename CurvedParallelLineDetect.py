import cv2
import numpy as np




def reparameterize_curve(pts, num_points=200):
    """Resample curve points uniformly in normalized arc length.


    Args:
        pts: Array of points defining the curve.
        num_points: Number of points in resampled curve.


    Returns:
        Array of uniformly spaced points along the curve.
    """
    dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    cumdist = np.concatenate(([0], np.cumsum(dists)))
    total_length = cumdist[-1]
    if total_length == 0:
        return pts
    t = cumdist / total_length
    t_uniform = np.linspace(0, 1, num_points)
    x_uniform = np.interp(t_uniform, t, pts[:, 0])
    y_uniform = np.interp(t_uniform, t, pts[:, 1])
    return np.vstack([x_uniform, y_uniform]).T




def process_frame(frame):
    """Process video frame to detect and draw parallel curves.


    Args:
        frame: Input video frame.


    Returns:
        Frame with detected curves and centerline drawn.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)


    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    max_area = frame.shape[0] * frame.shape[1] * 0.4  # 40% of frame area


    # Filter contours by area and aspect ratio
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
       
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h != 0 else 0
        if aspect_ratio > 0.1 and aspect_ratio < 10:  # Filter extreme aspect ratios
            filtered_contours.append(cnt)


    if len(filtered_contours) < 2:
        return frame
   
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
    line_contours = filtered_contours[:2]


    curves = []
    for cnt in line_contours:
        pts = cnt.reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(cnt)
        if w > h:
            pts_sorted = pts[np.argsort(pts[:, 0])]
        else:
            pts_sorted = pts[np.argsort(pts[:, 1])]
        curves.append(pts_sorted)


    num_points = 500
    curve1 = reparameterize_curve(curves[0], num_points)
    curve2 = reparameterize_curve(curves[1], num_points)
    center_line = (curve1 + curve2) / 2.0


    center_line_int = center_line.astype(np.int32).reshape((-1, 1, 2))
    curve1_int = curve1.astype(np.int32).reshape((-1, 1, 2))
    curve2_int = curve2.astype(np.int32).reshape((-1, 1, 2))


    cv2.polylines(frame, [center_line_int], isClosed=False, color=(0, 0, 255), thickness=6)
    cv2.polylines(frame, [curve1_int], isClosed=False, color=(255, 0, 0), thickness=2)
    cv2.polylines(frame, [curve2_int], isClosed=False, color=(0, 255, 255), thickness=2)
   
    return frame




def main():
    """Run the main video processing loop."""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        cv2.imshow("Processed Video", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()

