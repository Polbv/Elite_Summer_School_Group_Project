import cv2
import json

# Load your image
image_path = "media/raw/Picture_5.jpg"


DEPTH=False
if DEPTH:
    depth_path="media/raw/Depth_7.jpg" ##Optional for now
    depth_img = cv2.imread(depth_path)

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Store clicked circles here: [x, y, radius]
circles = []
radius = 20  # Default radius (can change in code or make interactive)
preview_center=None


# Mouse click callback
def draw_circle(event, x, y, flags, param):
    global img,depth_img, circles, preview_center, radius
    
    if event == cv2.EVENT_MOUSEWHEEL:  # Mouse wheel changes radius
        if flags > 0:
            radius += 1
        else:
            radius = max(1, radius - 1)
        preview_center = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:  # Move mouse to update preview
        preview_center = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to add
        if DEPTH:
            z=depth_img[y,x,1]

            circles.append((int(x), int(y),int(z), radius))
        else:
            circles.append((int(x), int(y), radius))
        cv2.circle(img, (x, y), radius, (0, 255, 0), 2)
        cv2.imshow("Image", img)
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right-click to remove last
        if circles:
            circles.pop()
            img[:] = original.copy()
            for cx, cy, _,r in circles:
                cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
            cv2.imshow("Image", img)

# Keep an original copy
original = img.copy()

# Open window and set callback
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", draw_circle)

print("Left-click to mark a circle center, right-click to remove last.")
print("Press 's' to save and quit, or 'q' to quit without saving.")

while True:
    display = img.copy()
    if preview_center:
        cv2.circle(display, preview_center, radius, (255, 0, 0), 1)  # preview in blue
    cv2.imshow("Image", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Save
        # Save to JSON
        with open("media/processed/labelled_image_circles.json", "w") as f:
            json.dump(circles, f, indent=4)
        print(f"Saved {len(circles)} circles to marked_circles.json")
        break
    elif key == ord('q'):
        print("Exiting without saving.")
        break

cv2.imwrite("media/processed/labelled_image.jpg", display)
cv2.destroyAllWindows()
