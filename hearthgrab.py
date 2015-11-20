# hearthgrab

import numpy as np
from PIL import ImageGrab
import cv2
import imutils
import glob
import os
import sys
import time
import win32gui, win32api, win32con
import math
import csv, json

MATCHING_METHODS = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
PROGRESS_BAR_LENGTH = 20
CARD_POSITIONS = [[171,159],[372,159],[573,159],[774,159],[171,476],[372,476],[573,476],[774,476]]
NEXT_PAGE_CLICK_POINT = [947,445]
CARD_SIZE = [174, 246]
TOTAL_NUMBER_OF_CARD_TEMPLATES = len(glob.glob(os.getcwd() + '/resources/card_templates/*'))
SIMILARITY_TOLERANCE = 10
PAGE_TURN_TIME = 0.3 #0.3
TWOEX_SIZE = [20,25]
CLASS_POSITION = [471,86]
CLASS_SIZE = [170,35]
PAGE_NO_POSITION = [500,750]
PAGE_NO_SIZE = [120,25]

def update_progress(progress, counter):
	sys.stdout.write("\r[{0}{1}] {2}% ({3})".format('#'*(int(math.ceil(progress*PROGRESS_BAR_LENGTH))),\
	' '*(int(math.floor((1 - progress)*PROGRESS_BAR_LENGTH))),\
	int(round(progress*100)),\
	counter))
	sys.stdout.flush()

### Import csv card data
card_data = {}
with open('resources/card_data.csv', 'rb') as f:
	reader = csv.reader(f, delimiter=',', quotechar='"')
	card_data = {rows[1]:[rows[0], rows[2], rows[3]] for rows in reader} # Name, ID, class, cost

### Load all template cards
print("> Loading all template cards")
CARD_TEMPLATES = [] 
CLASS_TEMPLATES = [] 
card_index = 0
start_time = time.clock()
for imagePath in glob.glob(os.getcwd() + '/resources/card_templates/*'):
	base = os.path.basename(imagePath)
	card_id = os.path.splitext(base)[0]
	CARD_TEMPLATES.append([card_id, cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)])
	card_index = card_index + 1
	update_progress(float(card_index) / float(TOTAL_NUMBER_OF_CARD_TEMPLATES), card_index)

class_names_in_order = ['druid', 'hunter', 'mage', 'paladin', 'priest', 'rogue', 'shaman', 'warlock', 'warrior', 'neutral']

for class_name in class_names_in_order:
	imagePath = os.getcwd() + '/resources/class_templates/' + class_name + '.png'
	CLASS_TEMPLATES.append([class_name, cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)])
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
	crop_img = im_numpy[win_y:win_h-110, win_x:win_w-450] #NOTE: its img[y: y + h, x: x + w] 
	gray_window = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	return gray_window

### Loop all pages
print("> Fetching all cards")	

start_time = time.clock()
grabbed_cards = []
current_class = None
page_count = 0
while (True):
	### Move the cursor so it does not trigger visual effects
	win32api.SetCursorPos((NEXT_PAGE_CLICK_POINT[0],NEXT_PAGE_CLICK_POINT[1]))
	
	### Take screenshot
	gray_window = takeScreenshot()

	### Find which class we are looking at #NOTE: its img[y: y + h, x: x + w] 
	class_image = gray_window[CLASS_POSITION[1]: CLASS_POSITION[1] + CLASS_SIZE[1], CLASS_POSITION[0]: CLASS_POSITION[0] + CLASS_SIZE[0]]
	for class_template in CLASS_TEMPLATES:
		class_result = cv2.matchTemplate(class_image, class_template[1], eval(MATCHING_METHODS[1]))
		(_, class_local_max_matching_value, _, _) = cv2.minMaxLoc(class_result)
		if (class_local_max_matching_value > 0.8):
			current_class = class_template[0]
			break
	
	### Find the cards on each page
	count = 0
	for card_position in CARD_POSITIONS:
		#card_image = gray_window[card_position[1]:card_position[1]+CARD_SIZE[1],card_position[0]:card_position[0]+CARD_SIZE[0]]
		card_image = gray_window[card_position[1]+110:card_position[1]+110+40,card_position[0]:card_position[0]+CARD_SIZE[0]]
		#cv2.imshow(str(count), card_image)
		x2_image = gray_window[card_position[1] + CARD_SIZE[1] : card_position[1] + CARD_SIZE[1] + TWOEX_SIZE[1],\
		card_position[0] + CARD_SIZE[0] / 2 - TWOEX_SIZE[0] : card_position[0] + CARD_SIZE[0] / 2 + TWOEX_SIZE[0]]
		count += 1
		#cv2.rectangle(gray_window, (CLASS_POSITION[0], CLASS_POSITION[1]), (CLASS_POSITION[0] + CLASS_SIZE[0], CLASS_POSITION[1] + CLASS_SIZE[1]), (0, 0, 255), 1)
		grabbed_cards.append([card_image, x2_image, current_class])
	#cv2.imshow("allcards", card_image)
	#cv2.waitKey(0)
	#sys.exit()
	
	### Click to the next page
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN|win32con.MOUSEEVENTF_ABSOLUTE,0,0)
	win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP|win32con.MOUSEEVENTF_ABSOLUTE,0,0)
	#win32api.ClipCursor((0,0,0,0))
	time.sleep (PAGE_TURN_TIME);
	page_count += 1
	
	### Check if we are on the last page
	new_window = takeScreenshot()
	
	last_window_page_no = gray_window[PAGE_NO_POSITION[1]: PAGE_NO_POSITION[1] + PAGE_NO_SIZE[1], PAGE_NO_POSITION[0]: PAGE_NO_POSITION[0] + PAGE_NO_SIZE[0]]
	next_window_page_no = new_window[PAGE_NO_POSITION[1]: PAGE_NO_POSITION[1] + PAGE_NO_SIZE[1], PAGE_NO_POSITION[0]: PAGE_NO_POSITION[0] + PAGE_NO_SIZE[0]]
	page_no_result = cv2.matchTemplate(last_window_page_no, next_window_page_no, eval(MATCHING_METHODS[1]))
	(_, max_page_matching_value, _, _) = cv2.minMaxLoc(page_no_result) 
	if (max_page_matching_value > 0.999):
		break

end_time = time.clock()
print("Collected all cards in %.2f seconds." % (end_time - start_time))
print("Scanned " + str(page_count) + " pages.")

### Save all found cards
# print("> Saving all found cards")
# start_time = time.clock()
# card_index = 0
# for card in grabbed_cards:
	# cv2.imwrite(os.getcwd() + '/resources/grabbed_cards/' + str(card_index) + '.png', card[0])
	# card_index = card_index + 1
	# update_progress(float(card_index) / float(len(grabbed_cards)), card_index)
# end_time = time.clock()
# print(" in %.2f seconds." % (end_time - start_time))

### Match all found cards
print("> Matching all found cards")

start_time = time.clock()
card_index = 0
last_matched_cost = 0
last_matched_class_name = ''
matched_cards = []
failed_cards = []
x2_template = cv2.cvtColor(cv2.imread('resources/x2.png'), cv2.COLOR_BGR2GRAY)
for card in grabbed_cards:
	tried_card_hearthpwn_id = None
	chosen_card_hearthpwn_id = None
	max_matching_value = 0
	current_class_name = card[2]
	template_crop_image = None
	
	if (current_class_name != last_matched_class_name):
		last_matched_cost = 0
	
	for template in CARD_TEMPLATES:
		tried_card_hearthpwn_id = template[0]
		template_class_name = card_data[tried_card_hearthpwn_id][1]
		if (template_class_name == ''):
			template_class_name = 'neutral'
		template_cost = card_data[tried_card_hearthpwn_id][2]
		
		if (template_cost < last_matched_cost):
			continue

		if (current_class_name == template_class_name.lower()):
			template_crop = template[1][118:118+24, 14:14+147]

			result = cv2.matchTemplate(card[0], template_crop, eval(MATCHING_METHODS[1]))
			(_, local_max_matching_value, _, _) = cv2.minMaxLoc(result)
			# if (template[0] == '12291'):
				# print local_max_matching_value
				# cv2.imshow(str(card_index), template_crop)
				# cv2.waitKey(0)
				# sys.exit()
			if (local_max_matching_value > max_matching_value):
				template_crop_image = template_crop
				max_matching_value = local_max_matching_value
				chosen_card_hearthpwn_id = tried_card_hearthpwn_id
	#print "\n" + str(tried_card_hearthpwn_id) + " - " + str(max_matching_value)
	
	if (max_matching_value > 0.45):
		#cv2.imshow(str(card_index), template_crop_image)
		x2_result = cv2.matchTemplate(card[1], x2_template, eval(MATCHING_METHODS[1]))
		(_, x2_local_max_matching_value, _, x2_maxLoc) = cv2.minMaxLoc(x2_result)
		has_x2 = x2_local_max_matching_value > 0.8
		
		card_count = 1
		if has_x2:
			card_count = 2
			
		match_data = {}
		match_data['count'] = card_count
		match_data['name'] = card_data[chosen_card_hearthpwn_id][0]
		#print match_data['name']
		match_data['golden'] = chosen_card_hearthpwn_id.endswith('-g')
		last_matched_class_name = card_data[chosen_card_hearthpwn_id][1]
		last_matched_cost = card_data[chosen_card_hearthpwn_id][2]
		matched_cards.append(match_data)
	else:
		failed_cards.append(card_index)
		#print "\n" + str(tried_card_hearthpwn_id) + " - " + str(max_matching_value)

	card_index = card_index + 1
	update_progress(float(card_index) / float(len(grabbed_cards)), card_index)
end_time = time.clock()
print(" in %.2f seconds." % (end_time - start_time))
print("Total " + str(len(matched_cards)) + " cards.")

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
