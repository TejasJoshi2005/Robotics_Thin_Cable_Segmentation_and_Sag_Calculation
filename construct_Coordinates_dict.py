import cv2
import os

# Put the path to your folder full of images here
IMG_FOLDER = "./Raw_Data/images" 

current_points = []

def click_event(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        # Increased circle radius to 15 so it's visible on high-res images
        cv2.circle(param['img'], (x, y), 15, (0, 255, 0), -1) 
        cv2.imshow("Annotation Tool", param['img'])

def run_picker():
    global current_points
    
    files = [f for f in os.listdir(IMG_FOLDER) if f.startswith('image_') and f.endswith('.jpg')]
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    print("DATASET_MANIFEST = {")
    
    # --- THE FIX: Configure the UI Window BEFORE the loop ---
    cv2.namedWindow("Annotation Tool", cv2.WINDOW_NORMAL)
    # Force the window to spawn at a comfortable 720p size (your OS will fit the image inside this)
    cv2.resizeWindow("Annotation Tool", 1280, 720) 
    
    for filename in files:
        img_path = os.path.join(IMG_FOLDER, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        current_points = []
        
        cv2.imshow("Annotation Tool", img)
        cv2.setMouseCallback("Annotation Tool", click_event, {'img': img})
        
        # Wait until the user clicks twice
        while len(current_points) < 2:
            key = cv2.waitKey(1) & 0xFF
            # Press 's' to skip an image if it's bad
            if key == ord('s'):
                print(f'    "{filename}": None, # Skipped')
                break
            # Press 'q' to quit early
            if key == ord('q'):
                cv2.destroyAllWindows()
                print("}")
                return

        if len(current_points) == 2:
            print(f'    "{filename}": [{current_points[0]}, {current_points[1]}],')

    print("}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_picker()