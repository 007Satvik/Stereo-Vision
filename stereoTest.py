import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
#Open both videos 
#Get the centre point as marked on the field, the centre of each frame, that is 
#the position of the camera. 
#hard code the height metric from metadata
#get the distance between the RCam and LCam
#Get rotations data using the grid lines on the field 
#make the essential matrix [R | t]
#Get tracked points from the algorithm
#trinagulation
#####  HARD CODE THESE TWO VARIABLE, MAKE SURE TO PUT DECIMAL ( . ) ########### 
marker_dist = 30.0 * (0.3048) #feet x (meters per feet)
Rheight = 16.0  #in meters
Lheight = 16.5  #in meters
##########################Global Variables######################################
Rpixel_location = []
Lpixel_location = []
Lyel_contours = []
Ryel_contours = []
Lcentroids = []
Rcentroids = []
pointsX = []
pointsY = []
pointsZ = []
Htranslation = ()
hScale = 0.0
HzScale = 0.0
rcam = cv2.VideoCapture("Right_trim.mp4")
lcam = cv2.VideoCapture("Left_trim.mp4")
Lcyan_min = Lmag_min = Lyel_min = Lk_min = 0
Lcyan_max = Lmag_max = Lyel_max = Lk_max = 255
Rcyan_min = Rmag_min = Ryel_min = Rk_min = 0
Rcyan_max = Rmag_max = Ryel_max = Rk_max = 255

######### Mouse Callback Functions ############################################
def Rmouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse clicked at (x={}, y={})".format(x, y))
        Rpixel_location.append((x, y))

def Lmouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse clicked at (x={}, y={})".format(x, y))
        Lpixel_location.append((x, y))

###############################################################################

################### BGR to CMYK conversion and mask ###########################
def Lapply_mask(image):
    global Lcyan_min, Lcyan_max, Lmag_min, Lmag_max, Lyel_min, Lyel_max
    global Lk_min, Lk_max
    c, m, y, k = cv2.split(image)
    mask = np.where((c <= Lcyan_max) & (c >= Lcyan_min) & 
                    (m <= Lmag_max) & (m >= Lmag_min) & 
                    (y <= Lyel_max) & (y >= Lyel_min) &
                    (k <= Lk_max) & (k >= Lk_min), 255, 0).astype(np.uint8)
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    return filtered_image, mask

def Rapply_mask(image):
    global Rcyan_min, Rcyan_max, Rmag_min, Rmag_max, Ryel_min, Ryel_max
    global Rk_min, Rk_max
    c, m, y, k = cv2.split(image)
    mask = np.where((c <= Rcyan_max) &( c >= Rcyan_min) & 
                    (m <= Rmag_max) & (m >= Rmag_min) &
                    (y <= Ryel_max) & (y >= Ryel_min) &
                    (k <= Rk_max) & (k >= Rk_min), 255, 0).astype(np.uint8)
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    return filtered_image, mask

def bgr_to_cmyk(image):
    b, g, r = cv2.split(image)

    # Normalize the channel values
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0

    k = 1 - np.maximum(np.maximum(r, g), b)
    c = (1 - r - k) / (1 - k + 1e-10)
    m = (1 - g - k) / (1 - k + 1e-10)
    y = (1 - b - k) / (1 - k + 1e-10)

    # Scale the values to the range [0, 255]
    k = (k * 255).astype(np.uint8)
    c = (c * 255).astype(np.uint8)
    m = (m * 255).astype(np.uint8)
    y = (y * 255).astype(np.uint8)

    cmyk_image = cv2.merge((c, m, y, k))
    return cmyk_image
###############################################################################

##################### Trackbar functions ######################################
#Left min
def trackbar_left_cyan_min(val):
    global Lcyan_max, Lcyan_min, left_first_frame
    Lcyan_min = val
    output_frame, _ = Lapply_mask(left_first_frame)
    cv2.imshow("Lmasked Frame", output_frame) 

def trackbar_left_magenta_min(val):
    global Lmag_max, Lmag_min, left_first_frame
    Lmag_min = val
    output_frame, _ = Lapply_mask(left_first_frame)
    cv2.imshow("Lmasked Frame", output_frame)

def trackbar_left_yellow_min(val):
    global Lyel_max, Lyel_min, left_first_frame
    Lyel_min = val
    output_frame, _ = Lapply_mask(left_first_frame)
    cv2.imshow("Lmasked Frame", output_frame)

def trackbar_left_k_min(val):
    global Lk_max, Lk_min, left_first_frame
    Lk_min = val
    output_frame, _ = Lapply_mask(left_first_frame)
    cv2.imshow("Lmasked Frame", output_frame)
#Left max 
def trackbar_left_cyan_max(val):
    global Lcyan_max, Lcyan_min, left_first_frame
    Lcyan_max = val
    output_frame, _ = Lapply_mask(left_first_frame)
    cv2.imshow("Lmasked Frame", output_frame) 

def trackbar_left_magenta_max(val):
    global Lmag_max, Lmag_min, left_first_frame
    Lmag_max = val
    output_frame, _ = Lapply_mask(left_first_frame)
    cv2.imshow("Lmasked Frame", output_frame)

def trackbar_left_yellow_max(val):
    global Lyel_max, Lyel_min, left_first_frame
    Lyel_max = val
    output_frame, _ = Lapply_mask(left_first_frame)
    cv2.imshow("Lmasked Frame", output_frame)

def trackbar_left_k_max(val):
    global Lk_max, Lk_min, left_first_frame
    Lk_max = val
    output_frame, _ = Lapply_mask(left_first_frame)
    cv2.imshow("Lmasked Frame", output_frame)

#Right min
def trackbar_right_cyan_min(val):
    global Rcyan_max, Rcyan_min, right_first_frame
    Rcyan_min = val
    output_frame, _ = Rapply_mask(right_first_frame)
    cv2.imshow("Rmasked Frame", output_frame) 

def trackbar_right_magenta_min(val):
    global Rmag_max, Rmag_min, right_first_frame
    Rmag_min = val
    output_frame, _ = Rapply_mask(right_first_frame)
    cv2.imshow("Rmasked Frame", output_frame)

def trackbar_right_yellow_min(val):
    global Ryel_max, Ryel_min, right_first_frame
    Ryel_min = val
    output_frame, _ = Rapply_mask(right_first_frame)
    cv2.imshow("Rmasked Frame", output_frame)

def trackbar_right_k_min(val):
    global Rk_max, Rk_min, right_first_frame
    Rk_min = val
    output_frame, _ = Rapply_mask(right_first_frame)
    cv2.imshow("Rmasked Frame", output_frame)
#right max 
def trackbar_right_cyan_max(val):
    global Rcyan_max, Rcyan_min, right_first_frame
    Rcyan_max = val
    output_frame, _ = Rapply_mask(right_first_frame)
    cv2.imshow("Rmasked Frame", output_frame) 
    cv2.waitKey(0)

def trackbar_right_magenta_max(val):
    global Rmag_max, Rmag_min, right_first_frame
    Rmag_max = val
    output_frame, _ = Rapply_mask(right_first_frame)
    cv2.imshow("Rmasked Frame", output_frame)

def trackbar_right_yellow_max(val):
    global Ryel_max, Ryel_min, right_first_frame
    Ryel_max = val
    output_frame, _ = Rapply_mask(right_first_frame)
    cv2.imshow("Rmasked Frame", output_frame)

def trackbar_right_k_max(val):
    global Rk_max, Rk_min, right_first_frame
    Rk_max = val
    output_frame, _ = Rapply_mask(right_first_frame)
    cv2.imshow("Rmasked Frame", output_frame)
###############################################################################
Rret, Rframe = rcam.read()
Lret, Lframe = lcam.read()
if not Rret or not Lret:
    print("Error loading the frames")
    exit(1)
left_first_frame = bgr_to_cmyk(Lframe)
right_first_frame = bgr_to_cmyk(Rframe)
#plt.imshow(left_first_frame)
cv2.namedWindow("trackbar")
def nothing(val):
    pass
#### Trackbars for min and max values of color filters ########################
def createTrackbars():
    cv2.createTrackbar("Lcyan_min", "trackbar", 0, 255, nothing)
    cv2.createTrackbar("Lmag_min", "trackbar", 0, 255, nothing)
    cv2.createTrackbar("Lyel_min", "trackbar", 0, 255, nothing)
    cv2.createTrackbar("Lk_min", "trackbar", 0, 255, nothing)
    cv2.createTrackbar("Lcyan_max", "trackbar", 255, 255, nothing)
    cv2.createTrackbar("Lmag_max", "trackbar", 255, 255, nothing)
    cv2.createTrackbar("Lyel_max", "trackbar", 255, 255, nothing)
    cv2.createTrackbar("Lk_max", "trackbar", 255, 255, nothing)

    cv2.createTrackbar("Rcyan_min", "trackbar", 0, 255, nothing)
    cv2.createTrackbar("Rmag_min", "trackbar", 0, 255, nothing)
    cv2.createTrackbar("Ryel_min", "trackbar", 0, 255, nothing)
    cv2.createTrackbar("Rk_min", "trackbar", 0, 255, nothing)
    cv2.createTrackbar("Rcyan_max", "trackbar", 255, 255, nothing)
    cv2.createTrackbar("Rmag_max", "trackbar", 255, 255, nothing)
    cv2.createTrackbar("Ryel_max", "trackbar", 255, 255, nothing)
    cv2.createTrackbar("Rk_max", "trackbar", 255, 255, nothing)
###############################################################################

######## Detect markers on field ##############################################
def detectMarkers():
    global Lyel_min, Lyel_max, Lk_max, Lk_min, Ryel_max, Ryel_min, Rk_max, Rk_min
    global Lyel_contours, Ryel_contours, Ryel_marker_max, Ryel_marker_min
    global Lyel_marker_max, Lyel_marker_min, Rk_marker_max, Rk_marker_min
    global Lk_marker_max, Lk_marker_min 
    global left_first_frame, right_first_frame
    while True:
        Lyel_min = cv2.getTrackbarPos("Lyel_min", "trackbar")
        Lyel_max = cv2.getTrackbarPos("Lyel_max", "trackbar")
        Lk_min   = cv2.getTrackbarPos("Lk_min", "trackbar")
        Lk_max   = cv2.getTrackbarPos("Lk_max", "trackbar")
        Ryel_max = cv2.getTrackbarPos("Ryel_max", "trackbar")
        Ryel_min = cv2.getTrackbarPos("Ryel_min", "trackbar")
        Rk_min   = cv2.getTrackbarPos("Rk_min", "trackbar")
        Rk_max   = cv2.getTrackbarPos("Rk_max", "trackbar")
        Lyel_output, Lyel_mask = Lapply_mask(left_first_frame)
        Ryel_output, Ryel_mask = Rapply_mask(right_first_frame)

        cv2.imshow("Left marker", Lyel_output)
        cv2.imshow("Right marker", Ryel_output)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('n'):
            break
    Ryel_marker_max = Ryel_max
    Ryel_marker_min = Ryel_min
    Rk_marker_max   = Rk_max
    Rk_marker_min   = Rk_min
    Lyel_marker_max = Lyel_max
    Lyel_marker_min = Lyel_min
    Lk_marker_max   = Lk_max
    Lk_marker_min   = Lk_min
    cv2.destroyWindow("Left marker")
    cv2.destroyWindow("Right marker")

    Lyel_contours, _ = cv2.findContours(Lyel_mask, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    Ryel_contours, _ = cv2.findContours(Ryel_mask, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)

def getMarkers_with_mouse():
    global Rframe, Lframe
    global Rpixel_location, Lpixel_location, angle, Htranslation
    cv2.namedWindow("rcam")
    cv2.namedWindow('lcam')
    cv2.setMouseCallback("rcam", Rmouse_callback)
    cv2.setMouseCallback("lcam", Lmouse_callback)
    while True:
        cv2.imshow("rcam", Rframe)
        cv2.imshow("lcam", Lframe)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        if len(Rpixel_location) > 2 and len(Lpixel_location) > 2:
            cv2.destroyWindow("rcam")
            cv2.destroyWindow("lcam")
            break
    #pixels have (x1, y1) , (x2, y2) -> atan2(y2 - y1 , x2 - x1)
    # Also, value of y increases when going downward, so subtract max y 
    # from all values of y
    print("Right (x1, y1), (x2, y2)")
    print("(" + str(Rpixel_location[0][0]) + ", " + str(Rframe.shape[:2][0] - Rpixel_location[0][1]) + "), (" + str(Rpixel_location[1][0]) + ", " + str(Rframe.shape[:2][0] - Rpixel_location[1][1]) + ")")
    print("Left (x1, y1), (x2, y2)")
    print("(" + str(Lpixel_location[0][0]) + ", " + str(Lpixel_location[0][1]) + "), (" + str(Lpixel_location[1][0]) + ", " + str(Lpixel_location[1][1]) + ")")
    # Rslope = ((Rframe.shape[:2][0] - Rpixel_location[1][1]) - 
    #           (Rframe.shape[:2][0] - Rpixel_location[0][1])) / (Rpixel_location[1][0] - Rpixel_location[0][0] + 1e-10)
    # Lslope = ((Lframe.shape[:2][0] - Lpixel_location[1][1]) - 
    #           (Lframe.shape[:2][0] - Lpixel_location[0][1])) / (Lpixel_location[1][0] - Lpixel_location[0][0] + 1e-10)
    inv = Rframe.shape[:2][0]
    Rvector = np.array([Rpixel_location[1][0] - Rpixel_location[0][0], (inv - Rpixel_location[0][1]) - (inv - Rpixel_location[1][1])])
    Lvector = np.array([Lpixel_location[1][0] - Lpixel_location[0][0], (inv - Lpixel_location[0][1]) - (inv - Lpixel_location[1][1])])
    newAngle = np.arccos(np.dot(Rvector, Lvector)/(np.linalg.norm(Rvector) * np.linalg.norm(Lvector)))
    Rangle = math.atan2(Rpixel_location[1][0] - Rpixel_location[0][0], 
                        Rpixel_location[1][1] - Rpixel_location[0][1])
    Langle = math.atan2(Lpixel_location[1][0] - Lpixel_location[0][0], 
                        Lpixel_location[1][1] - Lpixel_location[0][1])

    angle = Langle - Rangle
    #arctan = math.atan(abs(Rslope - Lslope / (1 + (Rslope * Lslope) + 1e-10)))
    #print("arctan")
    #print(arctan)
    print("Rangle: ")
    print(Rangle)
    print("Langle: ")
    print(Langle)
    print("Langle - Rangle")
    print(angle)
    print("New Angle")
    print(newAngle)
    while True:
        #show the points from right frame on left frame for comparision
        cv2.line(Lframe, (Lpixel_location[0][0], Lpixel_location[0][1]), 
             (Rpixel_location[1][0] - Rpixel_location[0][0] + Lpixel_location[0][0], Rpixel_location[1][1] - Rpixel_location[0][1] + Lpixel_location[0][1]), (0,0,255), 2)
        cv2.circle(Lframe, (Rpixel_location[2][0] - Rpixel_location[0][0] + Lpixel_location[0][0], Rpixel_location[2][1] - Rpixel_location[0][1] + Lpixel_location[0][1]), 10,(0,0,255), -1)
   
        cv2.line(Lframe, (Lpixel_location[0][0], Lpixel_location[0][1]), 
             (Lpixel_location[1][0], Lpixel_location[1][1]), (255,0,0), 2)
        cv2.circle(Lframe, Lpixel_location[2], 10,(255,0,0), -1)

        #cv2.imshow('Rresult', Rframe)
        cv2.imshow('Lresult', Lframe)
        key = cv2.waitKey(1)
        if  key == ord('q'):
            cv2.destroyWindow('Lresult')
            break
    
    Rcentre = (Rframe.shape[:2][1] // 2, Rframe.shape[:2][0] // 2)
    Lcentre = (Lframe.shape[:2][1] // 2, Lframe.shape[:2][0] // 2)
    RHtrans = [Rcentre[0] - Rpixel_location[2][0], Rcentre[1] - Rpixel_location[2][1]]
    LHtrans = [Lcentre[0] - Lpixel_location[2][0], Lcentre[1] - Lpixel_location[2][1]]
    Htranslation = [LHtrans[0] - RHtrans[0], LHtrans[1] - RHtrans[1]]
    print("Right pixel 2:")
    print(Rpixel_location[2])
    print("R Trans: ")
    print(RHtrans)
    print("L trans")
    print(LHtrans)
    print("Horizontal Translation ")
    print(Htranslation)
###############################################################################
def getFilters():
    global Lcyan_min, Lmag_min, Lyel_min, Lk_min, Lcyan_max, Lmag_max
    global Lyel_max, Lk_max, Rcyan_min, Rmag_min, Ryel_min, Rk_min
    global Rcyan_max, Rmag_max, Ryel_max, Rk_max
    while True:
        Lcyan_min = cv2.getTrackbarPos("Lcyan_min", "trackbar")
        Lmag_min = cv2.getTrackbarPos("Lmag_min", "trackbar")
        Lyel_min = cv2.getTrackbarPos("Lyel_min", "trackbar")
        Lk_min   = cv2.getTrackbarPos("Lk_min", "trackbar")
        Lcyan_max = cv2.getTrackbarPos("Lcyan_max", "trackbar")
        Lmag_max = cv2.getTrackbarPos("Lmag_max", "trackbar")
        Lyel_max = cv2.getTrackbarPos("Lyel_max", "trackbar")
        Lk_max   = cv2.getTrackbarPos("Lk_max", "trackbar")

        Rcyan_min = cv2.getTrackbarPos("Rcyan_min", "trackbar")
        Rmag_min = cv2.getTrackbarPos("Rmag_min", "trackbar")
        Ryel_min = cv2.getTrackbarPos("Ryel_min", "trackbar")
        Rk_min   = cv2.getTrackbarPos("Rk_min", "trackbar")
        Rcyan_max = cv2.getTrackbarPos("Rcyan_max", "trackbar")
        Rmag_max = cv2.getTrackbarPos("Rmag_max", "trackbar")
        Ryel_max = cv2.getTrackbarPos("Ryel_max", "trackbar")
        Rk_max   = cv2.getTrackbarPos("Rk_max", "trackbar")

        Loutput, _ = Lapply_mask(left_first_frame)
        Routput, _ = Rapply_mask(right_first_frame)

        cv2.imshow("Loutput", Loutput)
        cv2.imshow("Routput", Routput)
        #print(Lcyan_min)q
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break
        elif key == ord('q'):
            cv2.destroyWindow("Loutput")
            cv2.destroyWindow("Routput")
            break

def triangulate_points(P1, P2, points1, points2):
    # Convert image points to homogeneous coordinates
    print("inside tringulate, points1")
    print(points1)
    print("points2")
    print(points2)
    points1_homogeneous = cv2.convertPointsToHomogeneous(points1)
    points2_homogeneous = cv2.convertPointsToHomogeneous(points2)

    # Perform triangulation
    points_4d = cv2.triangulatePoints(P1, P2, points1_homogeneous, points2_homogeneous)

    # Convert 4D homogeneous coordinates to 3D Cartesian coordinates
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

    return points_3d.squeeze()

def heightScale():
    global P1, P2, hScale
    Rcentre = Rpixel_location[2]
    Lcentre = Lpixel_location[2]

    points_4d = cv2.triangulatePoints(P1, P2, Rcentre, Lcentre)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

    _, _, z = points_3d[0][0]
    hScale = Rheight / z

def horizontalScale():
    global HzScale
    rdistance = math.dist(Rpixel_location[0], Rpixel_location[1])
    HzScale = marker_dist/rdistance


createTrackbars()
getMarkers_with_mouse()
getFilters()

cv2.destroyAllWindows()
cos = np.cos(angle)
sin = np.sin(angle)

R = np.array([[cos, -sin, 0],
              [sin,  cos, 0],
              [0,      0, 1]])

T = np.array([[Htranslation[0]],
              [Htranslation[1]],
              [Lheight - Rheight]])

E = np.hstack((R, T))

print("R")
print(R)
print("T")
print(T)
print("E")
print(E)

Rmatrix = np.load("camera_matrix.npy")
print("Rmatrix")
print(Rmatrix)
Lmatrix = np.load("new_drone_camera_matrix.npy")
print("Lmatrix")
print(Lmatrix)

P1 = np.array(Rmatrix) @ np.hstack((np.eye(3), -np.zeros((3,1))))
P2 = np.array(Lmatrix) @ np.hstack((R, -T))

print("P1")
print(P1)
print("P2")
print(P2)

heightScale()
horizontalScale()
while True:
    Lret, left  = lcam.read()
    Rret, right = rcam.read()

    if not Lret:
        print("End of left video")
        break
    if not Rret:
        print("End of right video")
        break

    Loutput = bgr_to_cmyk(left)
    Routput = bgr_to_cmyk(right)

    Loutput, Lmask = Lapply_mask(Loutput)
    Routput, Rmask = Rapply_mask(Routput)

    Lcontours, _ = cv2.findContours(Lmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Rcontours, _ = cv2.findContours(Rmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for Lcontour in Lcontours:
        LM = cv2.moments(Lcontour)

        Lcx = int(LM['m10'] / (LM['m00'] + 1e-10))
        Lcy = int(LM['m01'] / (LM['m00'] + 1e-10))
        if(Lcx > 0):
            Lcentroids.append([Lcx, Lcy])

    for Rcontour in Rcontours:
        RM = cv2.moments(Rcontour)

        Rcx = int(RM['m10'] / (RM['m00'] + 1e-10))
        Rcy = int(RM['m01'] / (RM['m00'] + 1e-10))
        if(Rcx > 0):
            Rcentroids.append([Rcx, Rcy])

    if len(Lcentroids) == 0 or len(Rcentroids) == 0:
        continue
    Lcx, Lcy = np.mean(Lcentroids, axis=0)
    Rcx, Rcy = np.mean(Rcentroids, axis=0)
    Lcentroid = np.array([Lcx, Lcy], dtype=np.float32)
    Rcentroid = np.array([Rcx, Rcy], dtype=np.float32)
    print("L centroid")
    print(Lcentroid)
    print("R centroid")
    print(Rcentroid)

    cv2.circle(Loutput, (int(Lcx), int(Lcy)), 5, (0, 0, 225), -1)
    cv2.circle(Routput, (int(Rcx), int(Rcy)), 5, (0, 0, 225), -1)

    frame1 = cv2.resize(Loutput, (0, 0), fx=0.5, fy=0.5)
    frame2 = cv2.resize(Routput, (0, 0), fx=0.5, fy=0.5)

    combined_frame = cv2.hconcat([frame1, frame2])
    cv2.imshow('Combined Frames', combined_frame)
    #P1 is for right, P2 is for left
    points_4d = cv2.triangulatePoints(P1, P2, Rcentroid, Lcentroid)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

    x_3d, y_3d, z_3d = points_3d[0][0]
    print("[x , y, z]")
    print(points_3d[0][0])
    pointsX.append(x_3d * HzScale)
    pointsY.append(y_3d * HzScale)
    pointsZ.append(Rheight - (z_3d * hScale))



print(np.mean(pointsZ))
rcam.release()
lcam.release()
cv2.destroyAllWindows()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pointsX, pointsY, pointsZ)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# #Now we have angle of raotation of left camera wrt right camera, i.e., angle
# # try gettin the translationsal data
# Rcentre = (Rframe.shape[:2][1] // 2, Rframe.shape[:2][0] // 2)
# Lcentre = (Lframe.shape[:2][1] // 2, Lframe.shape[:2][0] // 2)
# print("Rcentre:")
# print(Rcentre)
# print("Lcentre:")
# print(Lcentre)
# Rdistance = (Rcentre[0] - Rpixel_location[2][0], 
#              Rcentre[1] - Rpixel_location[2][1])

# Ldistance = (Lcentre[0] - Lpixel_location[2][0], 
#              Lcentre[1] - Lpixel_location[2][1])
# print("Rdistance: ")
# print(Rdistance)
# print("Ldistance")
# print(Ldistance)


# # Load camera matrices, rotation, and translation matrices
# # P1 and P2 are the camera projection matrices for two views
# # points1 and points2 are the 2D points in image coordinates for the point you want to triangulate
# P1 = np.load('camera_matrix.npy')
# #P2 = np.load('P2.npy')
# points1 = np.array([[x1, y1]], dtype=np.float32)
# points2 = np.array([[x2, y2]], dtype=np.float32)

# # Perform triangulation
# points_3d = triangulate_points(P1, P2, points1, points2)

# # Extract 3D coordinates
# x_3d, y_3d, z_3d = points_3d[0]