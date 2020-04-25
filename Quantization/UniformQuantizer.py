import numpy as np
import cv2
#import ipcv
import os.path

def quantize(im, levels, qtype='uniform', maxCount=255, displayLevels=None):
    """
    Function to run uniform gray-level and improved gray-scale Quantization.
    This takes in an image, and buckets the gray values depending on the params.
    Args:
        im (array): image to be quantized as an array of values from 0 to 255
        levels (int): number of levels to quantize to.
            This should be a positive integer, and smaller than the maxCount.
        qtype (optional[string]): the type of quantization to perform.
            Can be either 'uniform' or 'igs'; Defaults to 'uniform'.
        maxCount (optional[int]): the maximum value for a digital count
        displayLevels (optional[int]): the number of gray levels to expand to.
            By default this value is None and will shrink the range of greys.
            This value should be a positive integer when provided.
    Return:
        the quantized image
    """
    # default value if we need to return early
    returnImage = im

    # get int type
    dtype = im.dtype

    if (displayLevels == None):
        # by default don't re-expand the image
        displayCount = levels
    elif displayLevels > 0:
        displayCount = displayLevels-1
    else:
        print("displayLevels is an invalid value")
        return returnImage

    # we're getting one more level than we should be, so minus 1
    if ((levels > 0) and (levels < maxCount)):
        levels = levels - 1
    else:
        print("levels needs to be a positive value, and smaller than the maxCount")
        return returnImage

    if (qtype == 'uniform'):
        # uniform method from lecture
        returnImage = np.floor((im/((maxCount+1)/float(levels))))*(displayCount/levels)

    elif (qtype == 'igs'):
        # error diffusion method from lecture

        # default error as 0 for the first pixel
        error = 0

        # the list of rows that will be turned into an image
        returnList = []
        for i in range(len(im)):
            returnRow = []
            for j in range(len(im[i])):
                # get a new digital count with the error
                errDC = im[i][j] + error
                # save the error for the next pixel
                error = errDC % (maxCount/levels)

                # calculate the new digital count, and append it to the row
                newDC = np.floor((errDC)/(maxCount/levels))
                returnRow.append(newDC*(displayCount/levels))
            # append the row to the final image
            returnList.append(np.array(returnRow))

        returnImage = np.array(returnList, dtype)

    else:
        # invalid qtype
        print('qtype is an invalid value, please use "uniform", or "igs"')

    print(returnImage)
    return np.array(returnImage, dtype)

im = cv2.imread('cameraman.tif', cv2.IMREAD_UNCHANGED)
#cv2.namedWindow('Originl Image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

numberLevels = 7
quantizedImage = quantize(im,
                            numberLevels,
                            qtype='uniform',
                            displayLevels=256)
#cv2.namedWindow(filename + ' (Uniform Quantization)', cv2.WINDOW_AUTOSIZE)
cv2.imshow(' (Uniform Quantization)', quantizedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

numberLevels = 7
quantizedImage = quantize(im,
                            numberLevels,
                            qtype='igs',
                            displayLevels=256)
#cv2.namedWindow(' (IGS Quantization)', cv2.WINDOW_AUTOSIZE)
cv2.imshow(' (IGS Quantization)', quantizedImage)
#cv2.imshow(' (Uniform Quantization)', quantizedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()