# import numpy as np
# import glob
 
# dir = "/home/atik666/py/figs/discs3"

# img_array = []
# for filename in glob.glob(dir+'/*.jpg'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)
 
 
# out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()

import cv2
import os

image_folder = "/home/atik666/py/figs/discs3"
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

""""""
import ffmpeg
(
    ffmpeg
    .input('/home/atik666/py/figs/discs3/*.png', pattern_type='glob', framerate=5)
    .output('/home/atik666/py/figs/discs3/movie.mp4')
    .run()
)



