import os
import sys
import cv2
import json

class ImageApp:
    def __init__(self, img_dir: str, resize_height: int, save_dir: str=os.getcwd()):
        self.img_dir = img_dir
        self.img = None
        self.resize_height = resize_height
        self.coordinates_data = {}
        self.coords = []
        self.rate = None
        self.window_name = 'image'
        cv2.namedWindow(self.window_name)
        cv2.moveWindow(self.window_name, 100, 100)

    def onMouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            org_x, org_y = int(x / self.rate), int(y / self.rate)
            self.coords.append((org_x, org_y))
            self.draw_coordinates(self.coords)
            cv2.imshow(self.window_name, self.img)
            print(self.coords)
    
    def draw_coordinates(self, coords):
        for coord in coords:
            x, y = coord
            x = int(x * self.rate)
            y = int(y * self.rate)
            cv2.circle(self.img, (x, y), 3, (0 ,0, 255), -1)

    def getResizeRate(self, height):
        return self.resize_height / height

    def resize(self, img):
        return cv2.resize(self.img, None, fx=self.rate, fy=self.rate)

    def run(self):
        img_list = os.listdir(self.img_dir)
        for img_name in img_list:
            print(f"Processing {img_name}")
            img_path = os.path.join(self.img_dir, img_name)
            self.img = cv2.imread(img_path)
            if self.img is None:
                if img_name.endswith('.mp4') or img_name.endswith('.MP4'):
                    cap = cv2.VideoCapture(img_path)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Cannot read {img_name}")
                        continue
                    self.img = frame
                else:
                    continue
            w, h = self.img.shape[:2]
            self.rate = self.getResizeRate(h)
            self.img = self.resize(self.img)
            cv2.imshow(self.window_name, self.img)
            cv2.setMouseCallback(self.window_name, self.onMouse)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    sys.exit("Quit")
                elif key == ord('c'):
                    self.coords = []
                    print(self.coords)
                elif key == ord('p'):
                    self.coords.pop()
                    print(self.coords)
                elif key == ord('s'):
                    self.coordinates_data[img_name] = self.coords.copy()
                    self.coords = []
                    break

            cv2.destroyAllWindows()
        print("All coordinates data collected")
        with open(os.path.join(os.path.expanduser(self.img_dir), 'subsets.json'), 'w') as f:
            json.dump(self.coordinates_data, f, indent=4)


if __name__ == '__main__':
    app = ImageApp(img_dir="cube", resize_height=800)
    app.run()
