import cv2
import numpy as np

# Load photo
photo = cv2.imread("media/raw/Template matching/pose_est_3.jpeg")


# Edge detection
# Edge detection

max_width = 800
max_height = 800
height, width = photo.shape[:2]
scaling_factor = min(max_width / width, max_height / height, 1)
if scaling_factor < 1:
    photo = cv2.resize(photo, (int(width * scaling_factor), int(height * scaling_factor)))

template = cv2.imread("media/raw/Template matching/Template.jpeg")  # actual PCB template
max_width = 800
max_height = 800
height, width = template.shape[:2]
scaling_factor = min(max_width / width, max_height / height, 1)
if scaling_factor < 1:
    template = cv2.resize(template, (int(width * scaling_factor), int(height * scaling_factor)))
def order_points(pts):
    # pts: array of shape (4,2)
    rect = np.zeros((4,2), dtype="float32")

    # Sum and diff help separate corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]       # bottom-right: largest sum
    rect[1] = pts[np.argmin(diff)]    # top-right: smallest difference
    rect[3] = pts[np.argmax(diff)]    # bottom-left: largest difference

    return rect
def find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Morphological closing (dilate then erode)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_img = gray.copy()
    # Find largest 4-point contour (rectangle)
    max_area = 0
    pcb_corners = None
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            print("here")
            color=(0,0,255)
            #color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.polylines(approx_img, [approx], isClosed=True, color=color, thickness=5)
            for x, y in approx.reshape(-1, 2):
                cv2.circle(approx_img, (x, y), 5, (0, 0, 255), -1)  # Draw corners

            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                pcb_corners = approx

    cv2.imshow("4-Point Polygons", approx_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pcb_corners
pcb_corners=find_corners(photo)

template_corners =find_corners(template)
if pcb_corners is None:
    print("PCB not found")
    exit()
for coords in pcb_corners:
        x=coords[0][0]
        y=coords[0][1]
        cv2.circle(photo, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red filled circle
        #cv2.putText(photo, f"{i+1}", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
for coords in template_corners:
        x=coords[0][0]
        y=coords[0][1]
        cv2.circle(template, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red filled circle

orb = cv2.ORB_create(5000)

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(template, None)
kp2, des2 = orb.detectAndCompute(photo, None)

# Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Match descriptors
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test (Lowe’s test)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
# template = cv2.resize(template, (pcb_w, pcb_h))
h, w = template.shape[:2]
# New corners after resize

# Convert detected corners to float32
# pcb_corners = order_points(np.float32(pcb_corners).reshape(-1,2))
# template_corners = order_points(np.float32(template_corners).reshape(-1,2))
if len(good_matches) >= 3:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,2)

# Warp template to photo
M,_ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
print(M)
warped_template = cv2.warpAffine(template, M, (photo.shape[1], photo.shape[0]))

# Overlay
overlay = cv2.addWeighted(photo, 0.7, warped_template, 0.3, 0)
overlay_2=cv2.addWeighted(photo, 0.3, warped_template, 0.7, 0)
#rescale image

# Show  
cv2.imshow("Aligned PCB", photo)
cv2.imshow("Aligned PCB 2", warped_template)
cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
