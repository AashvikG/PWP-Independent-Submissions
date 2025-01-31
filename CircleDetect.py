import cv2
import numpy as np
import time
def circle_detect(img):
  counter = 0
  # Convert to grayscale and blur
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  # Add a mask
  _, mask = cv2.threshold(gray, 105, 205, cv2.THRESH_BINARY)
  masked_image = cv2.bitwise_and(gray_blurred, gray_blurred, mask=mask)
  cv2.imshow("f", masked_image)
  # Detect edges
  edges = cv2.Canny(masked_image, 125, 175)
  cv2.imshow("f", edges)

  # Detect circles
  circles = cv2.HoughCircles(masked_image,
                             cv2.HOUGH_GRADIENT, 1, 50,
                             param1=150, param2=40,
                             minRadius=240, maxRadius=320)
  # Process single circle if found
  if circles is not None:
      circles = np.uint16(np.around(circles))
      # Only process first circle
      for pt in circles[0, :]:
          x, y, r = pt[0], pt[1], pt[2]
          # Draw the circumference of the circle
          cv2.circle(img, (x, y), r, (255, 0, 0), 3)
          # Draw center point
          cv2.circle(img, (x, y), 2, (0, 0, 255), 2)  # RED center point
          counter += 1
          if counter == 1:
              break
          
  return img


def main():
 # Capture webcam or video footage if needed
 cap = cv2.VideoCapture("IMG_0785.MOV")

 # Process Image
#    frame = cv2.imread("CANIMAGE.webp")
#    result = circle_detect(frame)
#    cv2.imshow("Circle Detection", result)


 if not cap.isOpened():
     print("Error: Could not open webcam.")
     return


 while True:
     # Uncomment to enable webcam mode
     # Run video processing
     ret, frame = cap.read()
     if not ret:
         break
     result = circle_detect(frame)
     cv2.imshow("Circle Detection", result)
     
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break


 cap.release()
 cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
