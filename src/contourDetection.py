import cv2
import numpy as np
import imutils

def find_countours(gray):
    '''
        Finds contour in the image

        Args:
            gray: homographic transormed binary image

        Return:
            im_out:  Hole filled binary image
            cnts:    list of detected countours
    '''
    
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    h, w = edged.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
 
    im_floodfill = edged.copy()
 
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = edged | im_floodfill_inv
    
    # find contours in the edge map
    cnts = cv2.findContours(im_out.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    return im_out, cnts


def find_best_contours_box(heatmap, threshold):

    '''
        Find the best high confident contours of the box

        Args:
            heatmap: homographic transormed binary image

        Return:
            countour: Hole filled binary image
            bbox:     best contour fitting box

    '''


    contour_area  = []
    contour_bbox  = []
    contour_angle = []
    
    # ret, heatmap   = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    countour, cnts = find_countours(heatmap)

    # return cnts
    for c in cnts:
       
        box  = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        contour_area.append(area)
        
        box  = [box[0], box[1], box[0]+box[2], box[1]+box[3]] 
        contour_bbox.append(box)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)

    print("con_count", len(cnts))
    # cnts_ = cnts[0] if len(cnts) == 2 else cnts[1]

    if len(cnts) == 0:
        return heatmap, None, None

    big_contour = max(cnts, key=cv2.contourArea)
    
    if len(contour_area)!=0:
        ind  = np.argmax(contour_area)
        cv2.drawContours(heatmap, cnts[ind], -1, (0, 255, 0), 5)
        return heatmap, contour_bbox[ind], cnts[ind]

    else:
        return None, None, None