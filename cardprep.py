import numpy as np
import cv2
import glob
import os
import sys
import time

CARD_SIZE = (174, 246)
#CARD_SIZE = (87, 123)
PROGRESS_BAR_LENGTH = 20
counter = 0

cardPaths = glob.glob(os.getcwd() + '/source_cards/*')
total_cards = len(cardPaths)

def update_progress(progress, counter):
	sys.stdout.write("\r[{0}{1}] {2}% ({3})".format('#'*(int(round(progress*PROGRESS_BAR_LENGTH))),\
	' '*(int(round((1 - progress)*PROGRESS_BAR_LENGTH))),\
	int(round(progress*100)),\
	counter))
	sys.stdout.flush()

if not os.path.exists(os.getcwd() + '/card_templates/'):
	os.makedirs(os.getcwd() + '/card_templates/')
	
start_time = time.clock()
for imagePath in glob.glob(os.getcwd() + '/source_cards/*'):
	im = cv2.imread(imagePath)
	base = os.path.basename(imagePath)
	cardname = os.path.splitext(base)[0]
	
	im_numpy = np.array(im)
	crop_img = im_numpy[40:344+40, 21:243+21] # NOTE: its img[y: y + h, x: x + w] 
	
	# ---- Image fingerprint transformation here
	gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	
	resized_img = cv2.resize(gray_img, CARD_SIZE)
	# ----
	
	cv2.imwrite(os.getcwd() + '/card_templates/' + cardname + '.png', resized_img)
	counter = counter + 1
	update_progress(float(counter) / float(total_cards), counter)
end_time = time.clock()
print(" in %.2f seconds." % (end_time - start_time))