Saving_frames_per_second = 30


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def extract_frames_from(video_file):
  # read the video file    
  cap = cv2.VideoCapture(video_file)

  # Check if camera opened successfully
  if(cap.isOpened()== False):
	  print("Error opening video stream or file")
  # get the FPS of the video
  fps = cap.get(cv2.CAP_PROP_FPS)
  # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
  saving_frames_per_second = min(fps, Saving_frames_per_second)
  # get the list of duration spots to save
  saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
  # start the loop
  count = 0
  frames = []
  while True:
      is_read, frame = cap.read()
      if not is_read:
          # break out of the loop if there are no frames to read
          break
      # get the duration by dividing the frame count by the FPS
      frame_duration = count / fps
      try:
          # get the earliest duration to save
          closest_duration = saving_frames_durations[0]
      except IndexError:
          # the list is empty, all duration frames were saved
          break
      if frame_duration >= closest_duration:
          # if closest duration is less than or equals the frame duration, 
          # then save the frame
          frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
          #cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame) 
          ## drop the duration spot from the list, since this duration spot is already saved
          frames.append(frame)
          try:
              saving_frames_durations.pop(0)
          except IndexError:
              pass
      # increment the frame count
      count += 1
  return frames


def opencv_to_pil_image(opencv_image):
  color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
  pil_image=Image.fromarray(color_coverted)
  return pil_image

def pil_to_opencv_image(pil_image):
  np_image=np.array(pil_image)  

  # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that 
  # the color is converted from RGB to BGR format
  opencv_image=cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
  return opencv_image

def opencv_images_to_video(opencv_images,video_filename):
  height, width, layers = opencv_images[0].shape
  image_size = (width, height)

  out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MP4V'), Saving_frames_per_second, image_size)
  for frame in opencv_images:
    out.write(frame)
  out.release()
