import sys
sys.path.insert(1,'src')
from detectionlib import inference_engine

def main():
  labels = 'face_labels.txt'
  interpret_im= inference_engine("models","images/2_Demonstration_Demonstration_Or_Protest_2_28_jpg.rf.d0ea6fc65e0327a7c8feea19cf64d608.jpg", "output_videos", "small", 10, 0.1, labels)
  interpret_im.run_image_detect('models/efficientdet-lite0_small_ax_face_300e_64b_edgetpu.tflite', 'output_image/image_processed.jpg', 'images/39_Ice_Skating_Ice_Skating_39_386_jpg.rf.369bdb65df2c454dbfbb9783072bf9c2.jpg', count=10 )

if __name__ == '__main__':
  main()
