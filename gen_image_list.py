from __future__ import print_function
import glob

# image_dir = '/home/gao/ClionProjects/ros_camera_publisher/src/depth_rgb_subscriber/build'
image_dir = '/home/gao/Desktop/image_saved_kuka_without_hand'
image_list = glob.glob(image_dir+'/'+'*.jpg')
output_image_list_path = image_dir + '/' + 'images.txt'
with open(output_image_list_path, 'w') as f:
    sorted_image_list = sorted(image_list)
    for path in sorted_image_list:
        print(path, file=f)
