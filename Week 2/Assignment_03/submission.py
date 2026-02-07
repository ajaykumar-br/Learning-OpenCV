import cv2

resizeUpDown = 0

def resizeTracker(*args):
    global resizeUpDown
    resizeUpDown = args[0]

def scaleUpDown(*args):
    if not resizeUpDown:
        scalingFactor = 1 + args[0]/100.0
    else:
        scalingFactor = 1 - args[0]/100.0
        if scalingFactor <= 0:
            scalingFactor = 0.01

    resizedImage = cv2.resize(img, None, fx = scalingFactor, fy = scalingFactor, interpolation=cv2.INTER_LINEAR)
    text = f"Mode: {'Scale Up' if resizeUpDown == 0 else 'Scale Down'} | Factor: {scalingFactor:.2f}x"
    
    cv2.putText(resizedImage, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Resize Image", resizedImage)


img = cv2.imread("truth.png", cv2.IMREAD_UNCHANGED)

cv2.namedWindow("Resize Image", cv2.WINDOW_AUTOSIZE) # checked other flags, but the results were same as the current flag or worse.
cv2.imshow("Resize Image", img)

cv2.createTrackbar("Scaling Factor", "Resize Image", 0, 100, scaleUpDown)
cv2.createTrackbar("Scale \n 0: ScaleUp \n 1: ScaleDown", "Resize Image", 0, 1, resizeTracker)
cv2.waitKey(0)
cv2.destroyAllWindows()