import numpy as np
import cv2
import glob
import os
import sys
import time
import math

CARD_SIZE = (174, 246)
PROGRESS_BAR_LENGTH = 20
counter = 0

cardPaths = glob.glob(os.getcwd() + '/resources/source_cards/*')
cardPaths += glob.glob(os.getcwd() + '/resources/source_cards_golden/*')
total_cards = len(cardPaths)

def update_progress(progress, counter):
	sys.stdout.write("\r[{0}{1}] {2}% ({3})".format('#'*(int(round(progress*PROGRESS_BAR_LENGTH))),\
	' '*(int(round((1 - progress)*PROGRESS_BAR_LENGTH))),\
	int(round(progress*100)),\
	counter))
	sys.stdout.flush()

if not os.path.exists(os.getcwd() + '/resources/card_templates_new/'):
	os.makedirs(os.getcwd() + '/resources/card_templates_new/')
	
cardPaths = [os.getcwd() + '/resources/source_cards/629.png']
start_time = time.clock()
for imagePath in cardPaths:
	#print imagePath
	img = cv2.imread(imagePath)
	base = os.path.basename(imagePath)
	cardname = os.path.splitext(base)[0]

	im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	has_white_background = im[2][2] > 125
	if (has_white_background):
		im_bordered = cv2.copyMakeBorder(im,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
		ret,thresh = cv2.threshold(im_bordered,230,255,0)
	else:
		im_bordered = cv2.copyMakeBorder(im,10,10,10,10,cv2.BORDER_CONSTANT,value=(0,0,0))
		ret,thresh = cv2.threshold(im_bordered,0,50,0)
	
	
	edges = cv2.Canny(thresh, 50, 100)
	_,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# cv2.imshow("0",edges)
	# cv2.waitKey(0)
	# sys.exit()
	areas = [cv2.contourArea(c) for c in contours]
	max_indexes = np.argsort(areas)[-2:][::-1]
	if (has_white_background):
		cnt = contours[max_indexes[1]]
	else:
		cnt = contours[max_indexes[0]]
	
	block_card = edges.copy()
	
	#cv2.drawContours(im_bordered, [cnt], -1, (0,255,0), 2)
	#cv2.drawContours(im, contours, -1, (0,255,0), 2)
	cv2.drawContours(block_card, [cnt], 0, (255,255,255), cv2.FILLED)
	card_edges = cv2.Canny(block_card, 50, 200)
	
	lines = cv2.HoughLinesP(card_edges, 1, math.pi/2, 2, None, 20, 1)
	flattened_lines = np.squeeze(lines)
	
	x_lines_to_remove = []
	for i in range (0, len(flattened_lines)):
		if (flattened_lines[i][0] == flattened_lines[i][2]):
			x_lines_to_remove.append(i)
	x_lines = np.delete(flattened_lines, x_lines_to_remove, 0)
	y_lines_to_remove = []
	for i in range (0, len(flattened_lines)):
		if (flattened_lines[i][1] == flattened_lines[i][3]):
			y_lines_to_remove.append(i)
	y_lines = np.delete(flattened_lines, y_lines_to_remove, 0)
	
	vertical_lines = y_lines[np.argsort(y_lines[:, 0])]
	vertical_lines_to_remove = []
	for i in range (0, len(vertical_lines)-1):
		if (abs(vertical_lines[i][0] - vertical_lines[i+1][0]) < 4):
			vertical_lines_to_remove.append(i)
	vertical_lines_purged = np.delete(vertical_lines, vertical_lines_to_remove, 0)

	horizontal_lines = x_lines[np.argsort(x_lines[:, 1])]
	horizontal_lines_to_remove = []
	for i in range (0, len(horizontal_lines)-1):
		#if (abs(horizontal_lines[i][1] - horizontal_lines[i+1][1]) < 4):
		horizontal_lines_to_remove.append(i)
	horizontal_lines_purged = np.delete(horizontal_lines, horizontal_lines_to_remove, 0)
	#print "removed:"
	#print horizontal_lines_purged

	#print vertical_lines
	x_limits = (vertical_lines_purged[0][0], vertical_lines_purged[-1][0])
	y_bottom = horizontal_lines_purged[0][1]
	y_limits = (int(round(y_bottom - float(x_limits[1] - x_limits[0]) * (345/float(244)))), y_bottom)
	#print y_limits
	#print horizontal_lines
	#x_limits = (x_left, x_right)
	#print x_limits
	
	# im_bordered = cv2.cvtColor(im_bordered, cv2.COLOR_GRAY2BGR)
	# for line in horizontal_lines_purged:
		# ### Find the relevant border lines
		# pt1 = (line[0],line[1])
		# pt2 = (line[2],line[3])
		# cv2.line(im_bordered, pt1, pt2, (0,0,255), 2)
	# for line in vertical_lines_purged:
		# ### Find the relevant border lines
		# pt1 = (line[0],line[1])
		# pt2 = (line[2],line[3])
		# cv2.line(im_bordered, pt1, pt2, (0,0,255), 2)
	# cv2.imshow("1",im_bordered)
	# cv2.waitKey(0)
	# sys.exit()
	
	crop_img = im_bordered[y_limits[0]+4+0:y_limits[1], x_limits[0]+0:x_limits[1]] 
	resized_img = cv2.resize(crop_img, CARD_SIZE)
	template_crop = resized_img[116:116+28, 12:12+151]
	
	# cv2.imshow("original",img)
	# cv2.imshow("resized",resized_img)
	# cv2.imshow("cropped",template_crop)
	# cv2.waitKey(0)
	
	cv2.imwrite(os.getcwd() + '/resources/card_templates_new/' + cardname + '.png', template_crop)
	counter = counter + 1
	update_progress(float(counter) / float(total_cards), counter)
end_time = time.clock()
print(" in %.2f seconds." % (end_time - start_time))