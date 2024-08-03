import cv2
import torch
from ultralytics import YOLO

class ArtworkDetector:
    def __init__(self, model):
        self.model = model
    
    def artwork_box(self, img):
        # Predict with the model
        results = self.model(img, verbose=False)  # predict on an image

        for result in results:
            boxes = result.boxes.data # Boxes object for bbox outputs
            boxes = boxes.to('cpu')
            boxes = boxes.numpy()

            max_conf = 0.0
            bbox = None
            for box in boxes:
                if ((box[-2]) > max_conf) and ((box[-2]) > 0.8):
                    bbox = [int(i) for i in box[:4]]
    
        if bbox!= None:
            return True, bbox

        elif bbox== None:
            return False, bbox

def load_yolo_model(artwork_model_path):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo_model = YOLO(artwork_model_path)
    yolo_model.to('cpu')
    return yolo_model

if __name__ == '__main__':
    img = cv2.imread('images/artwork-artwork-1-5AF96057-5C1D-4BCA-999A-405D1C9BA598-image.jpeg')
    artdetector = ArtworkDetector('weights/yolo_model_ver_2.pt')

    flag, bbox = artdetector.artwork_box(img)

    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255,0,0), thickness=4)
    
    cv2.imshow('title', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
