from net.Unet import Unet
import torch, rospy, cv2, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

import torch.backends.cudnn as cudnn

cv_bridge = CvBridge()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Available Device = {device}")
cudnn.enabled = True
model = Unet(class_num_=1, depth_=4, image_ch_=3, target_ch_=6).to(device)
frame = np.array([])
frame_rgb = np.array([])

def callback(img):
    process_image(img)


def process_image(img_msg):
    global frame
    global frame_rgb
    img = cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
    frame_rgb = img
    b,g,r = cv2.split(img)
    img = torch.Tensor(np.array([[b,g,r]])/255.0).to(device)

    with torch.no_grad():
        model.eval()
        start_ = time.time()
        output = model(img)
        result = np.zeros((output.shape[2], output.shape[3], 3), dtype=np.uint8)
        end_ = time.time()
        result[(output[0,0,:,:] >= 1.0).cpu()] = np.array([0,255,100])
        cv2.putText(result, f"Elapsed time : {end_-start_:.5f} sec", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,100))
        frame = result


if __name__ == "__main__":    
    data = torch.load("./carla_pth/unet_epoch050.pth")
    model.load_state_dict(data['model'])

    rospy.init_node('U-Net')
    sub_img = rospy.Subscriber('/carla/ego_vehicle/rgb_front/image', Image, callback, queue_size=1)

    while not rospy.is_shutdown():
        if not frame.shape[0]: continue
        frame = cv2.resize(frame, (500,500))
        frame_rgb = cv2.resize(frame_rgb, (500,500))
        cv2.imshow("seg", frame)
        cv2.imshow("rgb", frame_rgb)
        cv2.waitKey(1)
    rospy.spin()