
import cv2
import sympy
import time
import numpy as np
import imutils
from collections import deque

from rembg import remove, new_session
from src.find_artwork_bbox import ArtworkDetector
from src.imageSegmentor import Segment_Image
from src.contourDetection import find_best_contours_box

def appx_best_fit_ngon(contours, n = 6):
    # convex hull of the input mask
    # mask_cv2_gray = cv2.cvtColor(mask_cv2, cv2.COLOR_RGB2GRAY)
    # contours, _ = cv2.findContours(
    #     mask_cv2_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    # )
    hull = cv2.convexHull(contours)
    hull = np.array(hull).reshape((len(hull), 2))

    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to n vertices
    while len(hull) > n:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]

    return hull

def getInfo(x1, y1, x2, y2):
   return x1*y2 - y1*x2

def polyarea(points):
   N = len(points)
   firstx, firsty = points[0]
   prevx, prevy = firstx, firsty
   res = 0

   for i in range(1, N):
      nextx, nexty = points[i]
      res = res + getInfo(prevx,prevy,nextx,nexty)
      prevx = nextx
      prevy = nexty
   res = res + getInfo(prevx,prevy,firstx,firsty)
   return abs(res)/2.0

def simplify_contour(contour, n_corners=4):
    '''
    Binary searches best `epsilon` value to force contour 
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    '''
    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx

def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def segmentArt(img, yolo_model, sam_model, bg_session):
    
    h, w, _       = img.shape
    type          = 'bounding_box'

    threshold = 5
    yolo_whole_art = True

    flag, bbox_yolo    = ArtworkDetector(yolo_model).artwork_box(img)

    if bbox_yolo != None:
        area          = bbox_area(bbox_yolo)
        img_area      = h * w
        iou           = area / img_area

    else:
        iou = 1.0

    print("yolo iou",iou)
    # if the bounding box from yolo covers more than 97% of total picture then take whole image as mask
    if (iou > 0.97):
        
        bbox = [[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]
        type = 'bounding_box'
        mask = (np.ones((img.shape[0], img.shape[1]))*255).astype(np.uint8)

        print(type, bbox)
        return mask, type, bbox

    else:
        if bbox_yolo != None:
            margin   = 15
            x_low    = int(max(0, (bbox_yolo[1] - margin)))
            x_high   = int(min(img.shape[1], (bbox_yolo[3] + margin)))

            y_low    = int(max(0, (bbox_yolo[0] - margin)))
            y_high   = int(min(img.shape[0], (bbox_yolo[2] + margin)))

            img_crop = img[x_low:x_high, y_low:y_high]

        else:
            margin   = 15
            bbox_yolo = [0, 0, img.shape[1], img.shape[0]]

            x_low    = int(max(0, (bbox_yolo[1] - margin)))
            x_high   = int(min(img.shape[1], (bbox_yolo[3] + margin)))

            y_low    = int(max(0, (bbox_yolo[0] - margin)))
            y_high   = int(min(img.shape[0], (bbox_yolo[2] + margin)))

            img_crop = img.copy()

        # Use background Removal to find if the object is rectangular or irregular shape
        object_segmented         = remove(img_crop.copy(), session=bg_session)
        object_segmented         = cv2.cvtColor(object_segmented.copy(), cv2.COLOR_RGBA2BGR)

        countour_img             = cv2.cvtColor(object_segmented, cv2.COLOR_BGR2GRAY)
        ret, countour_img        = cv2.threshold(countour_img, threshold, 255, cv2.THRESH_BINARY)

        mask_cont                = (np.zeros((img.shape[0], img.shape[1]))).astype(np.uint8)
        mask_cont[x_low:x_high, y_low:y_high] = countour_img

        print('All shape',img.shape, mask_cont.shape, countour_img.shape)

        countour, bbox_contour, cnt      = find_best_contours_box(countour_img, threshold=threshold)
        approx  = simplify_contour(cnt) 

        length_box = [i[0] for i in approx[::-1]]

        print("num_points", len(length_box))
        area       = polyarea(length_box)
        cnt_area   = cv2.contourArea(cnt)
        
        iou        = area / cnt_area

        print('seg_iou', iou)

        # if intersection between polyarea and countour area is greater than 0.95 then it is rectangular nor it is irregular 
        if ((iou > 0.95) and (iou < 1.04) and (len(length_box)==4)):
            if bbox_yolo!=None:
                bbox = bbox_yolo.copy()              # if it is rectangular use yolo bounding box

            else:
                bbox_yolo = bbox_contour.copy()
                bbox      = bbox_contour.copy()

            type = 'bounding_box'

        else:
            if bbox_yolo!=None:
                bbox = bbox_yolo.copy()              # if it is rectangular use yolo bounding box

            else:
                bbox_yolo = bbox_contour.copy()
                bbox      = bbox_contour.copy()
            
            type = 'segmentation'

    bbox          = np.array(bbox)

    # img           = cv2.rectangle(img, (bbox[0], bbox[1]),(bbox[2], bbox[3]), color=(0,0,255), thickness=15)

    # Segment Anything
    start = time.time()
    mask, scores             = Segment_Image(sam_model, img, bbox)
    mask                     = (np.array(mask[0]).astype(np.float32)*255).astype(np.uint8)
    print("SAM_Time", time.time() - start)
    
    print("scores", scores[0])
    if scores[0]<=0.95:
        mask_cont_temp                   = np.zeros_like(mask_cont)
        countour, bbox_contour, cnt      = find_best_contours_box(mask_cont, threshold=threshold)
        mask_cont_temp = cv2.fillPoly(mask_cont_temp, pts =[cnt], color=(255,255,255))
        mask = mask_cont_temp.copy()

    else:
        mask_temp                   = np.zeros_like(mask)
        countour, bbox_contour, cnt      = find_best_contours_box(mask, threshold=threshold)
        mask_temp = cv2.fillPoly(mask_temp, pts =[cnt], color=(255,255,255))
        mask = mask_temp.copy()

        approx  = simplify_contour(cnt) 

        length_box = [i[0] for i in approx[::-1]]

        print("num_points", len(length_box))
        area       = polyarea(length_box)
        cnt_area   = cv2.contourArea(cnt)
        
        iou        = area / cnt_area

        print("iou SAM", iou)

        if ((iou > 0.95) and (iou < 1.04) and (len(length_box)==4)):
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=3)
            _, bbox_contour, cnt      = find_best_contours_box(mask, threshold=threshold)
            approx  = simplify_contour(cnt) 
            
            length_box = [i[0] for i in approx[::-1]]
            bbox       = np.array([list(i) for i in length_box])
            type = 'bounding_box'

            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, pts =[approx], color=(255,255,255))

        else:
            bbox = "None"
            type = 'segmentation'

    # mask   = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # img    = cv2.rectangle(img, (bbox_yolo[0], bbox_yolo[1]),(bbox_yolo[2], bbox_yolo[3]), color=(0,0,255), thickness=15)
    # mask   = cv2.rectangle(mask, (bbox_yolo[0], bbox_yolo[1]),(bbox_yolo[2], bbox_yolo[3]), color=(0,0,255), thickness=15)
    # concat = cv2.hconcat([img, mask])

    if type=='segmentation':
        bbox = "None"

    print(print(type, bbox))
    return mask, type, bbox

def load_bg_model(model):
    session = new_session(model)

    return session

def reposition_boxes_coordinates(bbox, angle):
    '''
        Rotate the coordiantes of the rectangular box
        Args:
            bbox: list of coordinates
            angle: Angle of box (Slope)

        Return:
            Rightward shifted list
    '''
    if abs(angle)>5.0:
        bbox = deque(bbox)
        bbox.rotate(1)

    return np.array(list(bbox))

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


def find_countours_and_segment_output(mask, source_img):
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = None
    max_area = 0
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area >= max_area: 
            max_area = cnt_area
            bbox     = cv2.boundingRect(cnt)

    x,y,w,h = bbox

    print(bbox)
    print(source_img.shape, thresh.shape)

    seg_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2BGRA)
    print(mask.max(), mask.min())
    seg_img[:,:,3] = mask
    return seg_img[y:y+h, x:x+w]
    

def find_homographic(bbox, source_img=None):

    bbox = np.array(bbox)
    print(bbox.shape)
    X =  bbox[:, 0]
    Y =  bbox[:, 1]

    indx = np.argsort(X)
    indy = np.argsort(Y)
    
    left_indx  = indx[:2]
    right_indx = indx[-2:]

    top_indx    = indy[:2]
    bottom_indx = indy[-2:]

    top_left     = bbox[list(set(left_indx).intersection(set(top_indx)))[0]]
    top_right    = bbox[list(set(right_indx).intersection(set(top_indx)))[0]]
    bottom_left  = bbox[list(set(left_indx).intersection(set(bottom_indx)))[0]]
    bottom_right = bbox[list(set(right_indx).intersection(set(bottom_indx)))[0]]

    source_pts        = np.array([top_left, top_right, bottom_right, bottom_left])
    width, height     = max(abs(source_pts[1][0] - source_pts[0][0]), abs(source_pts[2][0] - source_pts[3][0])), min(abs(source_pts[0][1] - source_pts[3][1]), abs(source_pts[1][1] - source_pts[2][1]))
    
    destination_pts   = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    print(source_pts, destination_pts,  width, height)
    # Compute the homography matrix
    H, _ = cv2.findHomography(source_pts, destination_pts)
    
    # Perform the warp perspective transformation
    warped_image = cv2.warpPerspective(source_img, H, (width, height))

    return warped_image


if __name__ == '__main__':
    bbox = [[726, 510],[169, 484], [171, 199], [730 ,75]]
    find_homographic(bbox)


