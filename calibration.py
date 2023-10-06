import cv2
import numpy as np
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# board = cv2.aruco.CharucoBoard((20, 15), 0.04, 0.02, aruco_dict)

# # Save the Charuco board as an image
# board_image = board.generateImage((1200, 900))  # Adjust the size of the board image as needed
# cv2.imwrite('charuco_calibration/charuco_board.png', board_image)


# Define the Charuco board parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((20, 15), 0.04, 0.02, aruco_dict)

# Arrays to store Charuco board corners and marker IDs from all calibration images
charuco_corners_list = []
charuco_ids_list = []

# Load and process calibration images
calibration_images = ["Left_drone(new)/DJI_0001.JPG", "Left_drone(new)/DJI_0002.JPG","Left_drone(new)/DJI_0003.JPG","Left_drone(new)/DJI_0004.JPG","Left_drone(new)/DJI_0005.JPG","Left_drone(new)/DJI_0006.JPG", ]  # List of calibration image file names
i = 0
for image_file in calibration_images:
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect Charuco board corners and marker IDs
    markers, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(markers) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markers, ids, gray, board)
        
        if ret:
            charuco_corners_list.append(charuco_corners)
            charuco_ids_list.append(charuco_ids)
            # cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids, (0, 255, 0))
            # cv2.aruco.drawDetectedMarkers(image, markers, ids, (0, 0, 255))
            # file_name = "charuco_calibration/detected_" + str(i) + '.png'
            # i = i + 1
            # cv2.imwrite(file_name, image)
            
    

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charuco_corners_list, charuco_ids_list, board, gray.shape[::-1], None, None)

# Display the camera matrix and distortion coefficients
print("RMSE value:")
print(ret)
print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)
np.save('new_drone_camera_matrix.npy', camera_matrix)
# Load the video file using OpenCV
cap = cv2.VideoCapture('Paper_Drone_Videos/120.9.MP4')

# Get the video properties (fps, width, height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter to save the undistorted frames as a new video
fourcc = cv2.VideoWriter_fourcc(*'avc1')
output_video = cv2.VideoWriter('undistorted_120.9.mp4', fourcc, fps, (width, height))

# Assuming you already have camera_matrix and dist_coeffs obtained from camera calibration
camera_matrix = np.array(camera_matrix)
dist_coeffs = np.array(dist_coeffs)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    #new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
    # Undistort the frame using the camera matrix and distortion coefficients
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None)
    # Write the undistorted frame to the output video
    #x, y, w, h = roi
    #undistorted = undistorted[y:y + h, x:x + w]
    output_video.write(undistorted)

    cv2.imshow('Undistorted Frame', undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()


