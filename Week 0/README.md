# Learning-OpenCV

## Day 1

-  Problems in Computer Vision
    - We typically solve one or more of these computer vision problems and package them in one neat application.
    - 1) Image Processing
        
        - Sub Problems like Image Denoising, enhancement and restoration. Image & Video Compression, Image Binarization and Binary image processing, Edge Detection.

    - 2) 3D reconstruction using 2D images, most common algorithm is "stereo vision"

        - Depth Mapping using 2 different images of the same scene from different viewpoints to extract 3D depth information.

        - Structure from Motion, Visual SLAM (Simultaneous Localization and Mapping). This is all extracting from motion information.

        - We can extract depth and shape information without moving the scene. It can be done by analysing the shading information, Photometric stereo uses 3 or more images of a scene with a static camera under different lighting conditions to obtain 3D shape information.

    - 3) Detecting & Matching Features in Geometric Computer Vision
        - Image alignment might mean if you want to see how a feature changes over multiple images.
    - 4) Recognition: 
        - Image Classification for one object in the image
        - For multiple objects, we use bounding box for detection.
        - Taking it forward for videos, we can do object tracking, but for that we need to know which object is same as the object in next frame.
        - Image Segmentation
        - Natural Image Mating (watch it again @ 8:30)
    
    - 5) Computational Photography (watch it again)