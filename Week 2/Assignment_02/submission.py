import cv2

# create boundingbox as selected by the user
def createAndCropBoundingBox(event, x, y, flags, param):
    global ix, iy, croppingSize

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(img, (ix, iy), (x,y), (255,0,255), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("Image", img)

        croppingSize = [(ix, iy), (x,y)]


# Read the required image
img = cv2.imread("sample.jpg", cv2.IMREAD_UNCHANGED)
dummy = img.copy()

# Create a namedWindow and se an event called Mouse Callback on that window
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", createAndCropBoundingBox)

# Set key to 0
k=0
# initial setting
ix, iy = -1, -1
croppingSize = None

# till ESC is not pressed, run
while(k != 27):
    cv2.imshow("Image", img)
    cv2.putText(img, "Choose top left corner, and drag.?", (2, 493), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 112, 50), 2, cv2.LINE_AA);

    # without this while loop will run infinitely and the image never appears
    k = cv2.waitKey(20) & 0xFF

    if croppingSize:
        x1, y1 = croppingSize[0]
        x2, y2 = croppingSize[1]

        cropped_img = dummy[y1:y2, x1:x2]
        cv2.imwrite("cropped_image.jpg", cropped_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()