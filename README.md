# OSP_final_project

This CPP project performs document scanning using the image of a document, often shaped as a trapezoid.

The input image to be processed should be located under the project folder.

When you run the code, you will be asked to enter the name of the input image file.
Type the name of the file and press enter. Note that you should specify the format of the image (e.g. jpg, png, etc).

After a while, you will be asked again to select whether you will save the output image as a grayscale or binary.
Enter 1 for grayscale, 2 for binary.

Then you can see the output popping up, also being saved under the project folder.


This cpp code uses OpenCV for image processing by the following procedure:

1. Edge detection
2. Boundary detection
3. Homography Transformation
4. Adaptive Thresholding

You can see the comments in the source code for the detailed implementation.
