import cv2

def main():
    # Read the image
    image = cv2.imread('image_blogs.png')
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Save the grayscale image
    cv2.imwrite('gray_image.png', gray_image)

if __name__ == "__main__":
    main()