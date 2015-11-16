# hearthgrab

import numpy as np
from PIL import ImageGrab
import cv2
import imutils
import glob
import os
import sys
import time
import win32gui
import win32api, win32con
import math
import csv, json

PROGRESS_BAR_LENGTH = 20
CARD_POSITIONS = [[171,159],[372,159],[573,159],[774,159],[171,476],[372,476],[573,476],[774,476]]
#CARD_POSITIONS = [[215,163],[415,163],[615,163],[815,163],[216,486],[416,486],[616,486],[816,486]]
NEXT_PAGE_CLICK_POINT = [947,445]
CARD_SIZE = [174, 246]
#CARD_SIZE = [87, 123]
TOTAL_NUMBER_OF_TEMPLATE_CARDS = len(glob.glob(os.getcwd() + '/cards/*'))
SIMILARITY_TOLERANCE = 10
PAGE_TURN_TIME = 0.3 #0.3

def update_progress(progress, counter):
	sys.stdout.write("\r[{0}{1}] {2}% ({3})".format('#'*(int(math.ceil(progress*PROGRESS_BAR_LENGTH))),\
	' '*(int(math.floor((1 - progress)*PROGRESS_BAR_LENGTH))),\
	int(round(progress*100)),\
	counter))
	sys.stdout.flush()

### Import csv card data
card_data = {}
with open('cardnames.csv', 'rb') as f:
	reader = csv.reader(f)
	card_data = {int(rows[0]):[rows[1], rows[2]] for rows in reader}

### Load all template cards
print("> Loading all template cards")
TEMPLATE_CARDS = [] ## TODO: Store cardname as index?
card_index = 0
start_time = time.clock()
for imagePath in glob.glob(os.getcwd() + '/cards/*'):
	base = os.path.basename(imagePath)
	cardname = os.path.splitext(base)[0]
	TEMPLATE_CARDS.append([cardname, cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)])
	card_index = card_index + 1
	update_progress(float(card_index) / float(TOTAL_NUMBER_OF_TEMPLATE_CARDS), card_index)
end_time = time.clock()
print(" in %.2f seconds." % (end_time - start_time))

### Find the Hearthstone client window and get its coordinates
print("> Getting window position")

def windowCallback(hwnd, position):
	if (win32gui.GetWindowText(hwnd) == 'Hearthstone'):
		#win32gui.BringWindowToTop(hwnd)
		rect = win32gui.GetWindowRect(hwnd)
		client_origo_on_screen = win32gui.ClientToScreen(hwnd, rect[:2])
		client_wh_on_screen = win32gui.ClientToScreen(hwnd, rect[2:])
		x = client_origo_on_screen[0] - rect[0]
		y = client_origo_on_screen[1] - rect[1]
		w = client_wh_on_screen[0] - client_origo_on_screen[0]
		h = client_wh_on_screen[1] - client_origo_on_screen[1]
		position.append(((x,y),(w,h)))
		NEXT_PAGE_CLICK_POINT[0] = NEXT_PAGE_CLICK_POINT[0] + x
		NEXT_PAGE_CLICK_POINT[1] = NEXT_PAGE_CLICK_POINT[1] + y

window_position = []
win32gui.EnumWindows(windowCallback, window_position)

if window_position is None:
	# Info for the user
	print "Open Hearthstone windowed, keep it in front and run this script."
	sys.exit()

### Take a screenshot and crop it to remove titlebar and border

def takeScreenshot():
	im = ImageGrab.grab()
	im_numpy = np.array(im)
	((win_x,win_y),(win_w,win_h)) = window_position[0]
	crop_img = im_numpy[win_y:win_h, win_x:win_w] #NOTE: its img[y: y + h, x: x + w] 
	gray_window = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	return gray_window

### Loop all pages
print("> Fetching all cards")	

start_time = time.clock()
grabbed_cards = []
TWOEX_SIZE = [20,25]
while (True):
	### Move the cursor so it does not trigger visual effects
	win32api.SetCursorPos((NEXT_PAGE_CLICK_POINT[0],NEXT_PAGE_CLICK_POINT[1]))
	
	### Take screenshot
	gray_window = takeScreenshot()
	
	### Find the cards on each page
	for card_position in CARD_POSITIONS:
		card_image = gray_window[card_position[1]:card_position[1]+CARD_SIZE[1],card_position[0]:card_position[0]+CARD_SIZE[0]]
		x2_image = gray_window[card_position[1] + CARD_SIZE[1] : card_position[1] + CARD_SIZE[1] + TWOEX_SIZE[1],\
		card_position[0] + CARD_SIZE[0] / 2 - TWOEX_SIZE[0] : card_position[0] + CARD_SIZE[0] / 2 + TWOEX_SIZE[0]]
		#cv2.rectangle(gray_window, (card_position[0], card_position[1]), (card_position[0] + CARD_SIZE[0], card_position[1] + CARD_SIZE[1]), (0, 0, 255), 1)
		#cv2.rectangle(gray_window, (card_position[0] + CARD_SIZE[0] / 2 - TWOEX_SIZE[0], card_position[1] + CARD_SIZE[1]),\
		#(card_position[0] + CARD_SIZE[0] / 2 + TWOEX_SIZE[0], card_position[1] + CARD_SIZE[1] + TWOEX_SIZE[1]), (0, 0, 255), 1)
		grabbed_cards.append([card_image, x2_image])
	#cv2.imshow("allcards", x2_image)
	#cv2.waitKey(0)
	#sys.exit()
	
	### Click to the next page
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN|win32con.MOUSEEVENTF_ABSOLUTE,0,0)
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP|win32con.MOUSEEVENTF_ABSOLUTE,0,0)
	#win32api.ClipCursor((0,0,0,0))
	time.sleep (PAGE_TURN_TIME);
	
	### Check if we are on the last page
	new_window = takeScreenshot()
	similar = np.allclose(gray_window, new_window, SIMILARITY_TOLERANCE)
	if (similar):
		break

end_time = time.clock()
print("Collected all cards in %.2f seconds." % (end_time - start_time))
print "Found: " + str(len(grabbed_cards))

### Save all found cards
print("> Saving all found cards")
start_time = time.clock()
card_index = 0
for card in grabbed_cards:
	cv2.imwrite(os.getcwd() + '/grabbed_cards/' + str(card_index) + '.png', card[0])
	card_index = card_index + 1
	update_progress(float(card_index) / float(len(grabbed_cards)), card_index)
end_time = time.clock()
print(" in %.2f seconds." % (end_time - start_time))

### Match all found cards
print("> Matching all found cards")

start_time = time.clock()
card_index = 0
matched_cards = []
failed_cards = []
x2_template = cv2.cvtColor(cv2.imread('x2.png'), cv2.COLOR_BGR2GRAY)
for card in grabbed_cards:
	template_index = 0
	card_hearthpwn_id = None
	max_val = 0
	
	for template in TEMPLATE_CARDS:
		#template_crop = template[4:CARD_SIZE[1] + 4,\
		#CARD_SIZE[0]/2 + 4:CARD_SIZE[0] + (CARD_SIZE[0]/2 + 4)]
		template_crop = template[1][10:CARD_SIZE[1], 10:CARD_SIZE[0]]
		# cv2.imshow(str(template_index), template_crop)
		# cv2.waitKey(0)
		# sys.exit()
		methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',\
		'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
		result = cv2.matchTemplate(card[0], template_crop, eval(methods[1]))
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
		
		if (maxVal > max_val):
			card_hearthpwn_id = template[0]
			max_val = maxVal
		template_index = template_index + 1
		
	if (max_val > 0.45):
		x2_result = cv2.matchTemplate(card[1], x2_template, eval(methods[1]))
		(_, x2_maxVal, _, x2_maxLoc) = cv2.minMaxLoc(x2_result)
		has_x2 = x2_maxVal > 0.8
		
		card_count = 1
		if has_x2:
			card_count = 2
			
		match_data = {}
		match_data['count'] = card_count
		match_data['name'] = card_data[int(card_hearthpwn_id)][0]
		#print str(len(card_data)) + " card data"
		matched_cards.append(match_data)
	else:
		failed_cards.append(card_index)
		#print "max: " + str(max_val)

	card_index = card_index + 1
	update_progress(float(card_index) / float(len(grabbed_cards)), card_index)
end_time = time.clock()
print(" in %.2f seconds." % (end_time - start_time))

### Save cards to file
print("> Saving card data to hearthgrab.txt")

with open('hearthgrab.txt', 'w') as outfile:
    json.dump(matched_cards, outfile)

#print "failed:"
#print failed_cards
#print matched_cards
#cv2.imshow("test", edged_screen)
#cv2.waitKey(0)
#sys.exit()
