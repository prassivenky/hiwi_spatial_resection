import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image.
img_raw = cv2.imread(r'C:\Users\VWTGHJC.VW\OneDrive - Volkswagen AG\Prassi\LUH Submissions\New_Tasks\frame000676.png', cv2.IMREAD_COLOR)

# N = Number of GCPs (=6)
# The offset is to just control the size of the ROIs around the GCPs
N = 6
offset = 10

centre_coords = []

# To crop an ROI around every GCP in the raw image and then detect the image coordinate
x_coords = [(1770,1920), (1665,1815),(1175,1325),(825,975),(1175,1325),(1450,1600)]
y_coords = [(805,955),(550,700),(385,535), (920,1070),(1360,1510),(1265,1415)]

# Global coordinates of 100, 200, 300, 400, 500 and 600 point IDs
gcp1 = [3847035.623, 653925.637, 5028341.286]
gcp2 = [3847040.443, 653932.336, 5028336.695]
gcp3 = [3847038.332, 653948.959, 5028336.034]
gcp4 = [3847021.776, 653953.108, 5028348.178]
gcp5 = [3847015.17, 653936.559, 5028355.386]
gcp6 = [3847021.019, 653929.275, 5028351.63]
gcps = [gcp1,gcp2,gcp3,gcp4,gcp5,gcp6]

for i in range(N):
    
    roi_rows = [y_coords[i][0]+offset, y_coords[i][1]-offset]
    
    roi_cols = [x_coords[i][0]+offset, x_coords[i][1]-offset]
    
    img1 = img_raw[roi_rows[0]:roi_rows[1],roi_cols[0]:roi_cols[1]]

    output = img_raw.copy()
    
    # Convert to grayscale.
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    # Median Blur using 3 * 3 kernel.
    gray_blurred = cv2.medianBlur(gray, 5)
    
    # Applying Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
    				cv2.HOUGH_GRADIENT, 1, 20, param1 = 100,
    			param2 = 30, minRadius = 5, maxRadius = 75)
    
    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0,:]:        
            a, b, r = pt[0], pt[1], pt[2]

    gcp_centre = (a+roi_cols[0], b+roi_rows[0])
    print(r'The image coordinates of this GCP are (x,y) = '+str(gcp_centre))
    cv2.circle(img_raw, gcp_centre, r, (0,0,255), 3)
    cv2.circle(img_raw, gcp_centre, 1, (0,0,255), 3)
    centre_coords.append(gcp_centre)

    cv2. namedWindow("Window", cv2.WINDOW_KEEPRATIO) 
    cv2.imshow("Window", img_raw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Assembling the 12x12 design matrix --> 6 GCPs    
A = np.zeros((12,12))
templist = []
for i in range(N):
    temp1 = np.array([gcps[i][0], gcps[i][1], gcps[i][2],1,0,0,0,0,-centre_coords[i][0]*gcps[i][0],-centre_coords[i][0]*gcps[i][1],-centre_coords[i][0]*gcps[i][2],-centre_coords[i][0]])
    temp2 = np.array([0,0,0,0,gcps[i][0], gcps[i][1], gcps[i][2],1,-centre_coords[i][1]*gcps[i][0],-centre_coords[i][1]*gcps[i][1],-centre_coords[i][1]*gcps[i][2],-centre_coords[i][1]])
    templist.append(temp1)
    templist.append(temp2)
    
A = np.vstack(templist)

M = A.T @ A

# Choosing the eigenvector corresponding the smallest eigenvalue and reshaping it
val, vec = np.linalg.eig(M)
idx = np.argsort(val)
val = val[idx]
vec = vec[:,idx]
m = vec[:,0].reshape((3,4))
m_dof = m/m[-1,-1]

# The intrinsic matrix of the camera (or the camera matrix) directly from Seafile.
cam_int = [ -1.2960615355458765e+03, 0., 1.2943812632671218e+03, 0.,
       -1.2957562700930018e+03, 1.0520247354104461e+03, 0., 0., 1. ]

cam_mat = np.array(cam_int).reshape((3,3))

cam_ext = np.linalg.inv(cam_mat) @ m_dof


translation = cam_ext[:,-1]
rotation = cam_ext[:,0:-1]

    
