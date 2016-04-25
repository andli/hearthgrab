# hearthgrab

import csv
import glob
import json
import math
import os
import sys
import time
import win32api
import win32con
import win32gui
from time import sleep

import cv2
import numpy as np
from PIL import ImageGrab


# Help methods
def update_progress(progress, counter):
    sys.stdout.write("\r[{0}{1}] {2}% ({3})".format('#' * (int(math.ceil(progress * PROGRESS_BAR_LENGTH))),
                                                    ' ' * (int(math.floor((1 - progress) * PROGRESS_BAR_LENGTH))),
                                                    int(round(progress * 100)),
                                                    counter))
    sys.stdout.flush()


MATCHING_METHODS = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
                    'cv2.TM_SQDIFF_NORMED']
PROGRESS_BAR_LENGTH = 20
#CARD_POSITIONS = [[171, 159], [372, 159], [573, 159], [774, 159], [171, 476], [372, 476], [573, 476], [774, 476]]
#CARD_SIZE = [174, 246]
#TOTAL_NUMBER_OF_CARD_TEMPLATES = len(glob.glob(os.getcwd() + '/resources/card_templates/*'))
#SIMILARITY_TOLERANCE = 10
PAGE_TURN_TIME = 0.3  # 0.3
TWO_X_SIZE = [20, 25]  # TODO: Make relative or find in other way
CLASS_POSITION = [471, 86]  # TODO: Make relative or find in other way
CLASS_SIZE = [170, 35]
PAGE_NO_POSITION = [500, 750]
PAGE_NO_SIZE = [120, 25]
CLASS_NAMES_IN_ORDER = ['druid', 'hunter', 'mage', 'paladin', 'priest', 'rogue', 'shaman', 'warlock', 'warrior',
                        'neutral']
CLASS_TEMPLATES = []

# Determined constants
WINDOW_RECTANGLE = []
CARD_PAGE_RECTANGLE = []
NEXT_PAGE_CLICK_POINT = []


for class_name in CLASS_NAMES_IN_ORDER:
    imagePath = os.getcwd() + '/resources/class_templates/' + class_name + '.png'
    CLASS_TEMPLATES.append([class_name, cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)])

# Import csv card data
with open('resources/card_data.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    card_data = {rows[1]: [rows[0], rows[2], rows[3]] for rows in reader}  # Name, ID, class, cost

# Find the Hearthstone client window and get its coordinates
print("> Getting window position")


def window_callback(hwnd, w_rectangle):
    if win32gui.GetWindowText(hwnd) == 'Hearthstone':
        rect = win32gui.GetWindowRect(hwnd)
        client_origo_on_screen = win32gui.ClientToScreen(hwnd, rect[:2])
        client_wh_on_screen = win32gui.ClientToScreen(hwnd, rect[2:])
        x = client_origo_on_screen[0] - rect[0]
        y = client_origo_on_screen[1] - rect[1]
        w = client_wh_on_screen[0] - client_origo_on_screen[0]
        h = client_wh_on_screen[1] - client_origo_on_screen[1]
        w_rectangle.append(((x, y), (w, h)))


win32gui.EnumWindows(window_callback, WINDOW_RECTANGLE)
hs_hwnd = win32gui.FindWindow(None, 'Hearthstone')
win32gui.ShowWindow(hs_hwnd, 5)
win32gui.SetForegroundWindow(hs_hwnd)
sleep(1)


# Take a screenshot and crop it to remove titlebar and border
def screenshot_and_crop_to_window():
    im = ImageGrab.grab()
    im_numpy = np.array(im)
    ((win_x, win_y), (win_w, win_h)) = WINDOW_RECTANGLE[0]
    cropped_window_image = im_numpy[win_y:win_y + win_h, win_x:win_x + win_w]  # NOTE: its img[y: y + h, x: x + w]
    return cropped_window_image


def find_card_page(cropped_window_image):
    gray_window = cv2.cvtColor(cropped_window_image, cv2.COLOR_BGR2GRAY)

    # Find card area
    ret, thresh = cv2.threshold(gray_window, 128, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)
    ((win_x, win_y), (win_w, win_h)) = WINDOW_RECTANGLE[0]
    return x + win_x, y + win_y, w, h


# Save click point
def get_click_point():
    cropped_img = screenshot_and_crop_to_window()
    x, y, w, h = find_card_page(cropped_img)
    return x + w - 20, y + w / 2


def get_card_page_image(cropped_window_image, card_page_bounding_rect):
    x, y, w, h = card_page_bounding_rect
    crop_cards = cropped_window_image[y:y + h, x:x + w]

    # Reverse BGR
    crop_cards = cv2.cvtColor(crop_cards, cv2.COLOR_BGR2RGB)
    # cv2.imshow("allcards", crop_cards)
    # cv2.waitKey(0)
    # sys.exit()
    return crop_cards


if WINDOW_RECTANGLE is None:
    # Info for the user
    print "Open Hearthstone windowed, keep it in front and run this script."
    sys.exit()

print("Found! x,y: " + str(WINDOW_RECTANGLE[0][0]) + ", size: " + str(WINDOW_RECTANGLE[0][1]))

NEXT_PAGE_CLICK_POINT = get_click_point()
CARD_PAGE_RECTANGLE = find_card_page(screenshot_and_crop_to_window())


# Take a screenshot and crop it to the card page
def screenshot_and_crop_to_card_page():
    im = ImageGrab.grab()
    im_numpy = np.array(im)
    x, y, w, h = CARD_PAGE_RECTANGLE
    cropped_card_page_image = im_numpy[y:y + h, x:x + w]  # NOTE: its img[y: y + h, x: x + w]
    crop_cards = cv2.cvtColor(cropped_card_page_image, cv2.COLOR_BGR2RGB)
    return crop_cards


# Loop all pages
print("> Looping all pages")

start_time = time.clock()
grabbed_cards = []
current_class = None
page_count = 0
while True:
    # Move the cursor so it does not trigger visual effects
    win32api.SetCursorPos((NEXT_PAGE_CLICK_POINT[0], NEXT_PAGE_CLICK_POINT[1]))

    # Take screenshot
    cards_area = screenshot_and_crop_to_card_page()
    cv2.imshow("allcards", cards_area)
    cv2.waitKey(0)
    sys.exit()

    # ------------ here, get text areas

    # # Find the cards on each page
    # count = 0
    # for card_position in CARD_POSITIONS:
    #     # card_image = gray_window[card_position[1]:card_position[1]+CARD_SIZE[1],card_position[0]:card_position[0]+CARD_SIZE[0]]
    #     card_image = cards_area[card_position[1] + 110:card_position[1] + 110 + 40,
    #                  card_position[0]:card_position[0] + CARD_SIZE[0]]
    #     # cv2.imshow(str(count), card_image)
    #     x2_image = cards_area[card_position[1] + CARD_SIZE[1]: card_position[1] + CARD_SIZE[1] + TWO_X_SIZE[1], \
    #                card_position[0] + CARD_SIZE[0] / 2 - TWO_X_SIZE[0]: card_position[0] + CARD_SIZE[0] / 2 +
    #                                                                     TWO_X_SIZE[0]]
    #     count += 1
    #     # cv2.rectangle(gray_window, (CLASS_POSITION[0], CLASS_POSITION[1]), (CLASS_POSITION[0] + CLASS_SIZE[0], CLASS_POSITION[1] + CLASS_SIZE[1]), (0, 0, 255), 1)
    #     grabbed_cards.append([card_image, x2_image, current_class])
    # # cv2.imshow("allcards", card_image)
    # # cv2.waitKey(0)
    # # sys.exit()
    #
    # # Click to the next page
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_ABSOLUTE, 0, 0)
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP | win32con.MOUSEEVENTF_ABSOLUTE, 0, 0)
    # # win32api.ClipCursor((0,0,0,0))
    # time.sleep(PAGE_TURN_TIME);
    # page_count += 1
    #
    # # Check if we are on the last page
    # new_window = screenshot_and_crop_to_card_page()
    #
    # last_window_page_no = cards_area[PAGE_NO_POSITION[1]: PAGE_NO_POSITION[1] + PAGE_NO_SIZE[1],
    #                       PAGE_NO_POSITION[0]: PAGE_NO_POSITION[0] + PAGE_NO_SIZE[0]]
    # next_window_page_no = new_window[PAGE_NO_POSITION[1]: PAGE_NO_POSITION[1] + PAGE_NO_SIZE[1],
    #                       PAGE_NO_POSITION[0]: PAGE_NO_POSITION[0] + PAGE_NO_SIZE[0]]
    # page_no_result = cv2.matchTemplate(last_window_page_no, next_window_page_no, eval(MATCHING_METHODS[1]))
    # (_, max_page_matching_value, _, _) = cv2.minMaxLoc(page_no_result)
    # if (max_page_matching_value > 0.999):
    #     break
    break

end_time = time.clock()
print("Collected all cards in %.2f seconds." % (end_time - start_time))
print("Scanned " + str(page_count) + " pages.")

# Save all found cards
print("> Saving all found cards")
start_time = time.clock()
card_index = 0
for card in grabbed_cards:
    cv2.imwrite(os.getcwd() + '/resources/grabbed_cards/' + str(card_index) + '.png', card[0])
    card_index += 1
    update_progress(float(card_index) / float(len(grabbed_cards)), card_index)
end_time = time.clock()
print(" in %.2f seconds." % (end_time - start_time))

# Match all found cards
print("> Matching all found cards")

start_time = time.clock()
card_index = 0
last_matched_cost = 0
last_matched_class_name = ''
matched_cards = []
failed_cards = []
number_of_blanks = 0
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
            result = cv2.matchTemplate(card[0], template[1], eval(MATCHING_METHODS[1]))
            (_, local_max_matching_value, _, _) = cv2.minMaxLoc(result)
            # if (template[0] == '12291'):
            # print local_max_matching_value
            # cv2.imshow(str(card_index), template[1])
            # cv2.waitKey(0)
            # sys.exit()
            if (local_max_matching_value > max_matching_value):
                template_crop_image = template[1]
                max_matching_value = local_max_matching_value
                chosen_card_hearthpwn_id = tried_card_hearthpwn_id
    # print "\n" + str(card_data[chosen_card_hearthpwn_id][0]) + " - " + str(max_matching_value)

    if (max_matching_value > 0.70):
        # cv2.imshow(str(card_index), template_crop_image)
        x2_result = cv2.matchTemplate(card[1], x2_template, eval(MATCHING_METHODS[1]))
        (_, x2_local_max_matching_value, _, x2_maxLoc) = cv2.minMaxLoc(x2_result)
        has_x2 = x2_local_max_matching_value > 0.8

        card_count = 1
        if has_x2:
            card_count = 2

        match_data = {}
        match_data['count'] = card_count
        match_data['name'] = card_data[chosen_card_hearthpwn_id][0]
        # print match_data['name']
        match_data['golden'] = chosen_card_hearthpwn_id.endswith('-g')
        match_data['match_rate'] = int(round(max_matching_value * 100))
        last_matched_class_name = card_data[chosen_card_hearthpwn_id][1]
        last_matched_cost = card_data[chosen_card_hearthpwn_id][2]
        matched_cards.append(match_data)
    elif (max_matching_value > 0.30):
        failed_cards.append([card_index, card_data[chosen_card_hearthpwn_id][0], max_matching_value])
        if not os.path.exists(os.getcwd() + '/failed_cards/'):
            os.makedirs(os.getcwd() + '/failed_cards/')
        cv2.imwrite(os.getcwd() + '/failed_cards/' + str(card_index) + '-card-' + card_data[chosen_card_hearthpwn_id][
            0] + '.png', card[0])
        cv2.imwrite(os.getcwd() + '/failed_cards/' + str(card_index) + '-temp-' + card_data[chosen_card_hearthpwn_id][
            0] + '.png', template_crop_image)
    # print "\n" + str(tried_card_hearthpwn_id) + " - " + str(max_matching_value)
    else:
        number_of_blanks += 1
    card_index = card_index + 1
    update_progress(float(card_index) / float(len(grabbed_cards)), card_index)
end_time = time.clock()
print(" in %.2f seconds." % (end_time - start_time))
print("Total " + str(len(matched_cards)) + " cards.")

# Save cards to file
print("> Saving card data to hearthgrab.txt")

with open('hearthgrab.txt', 'w') as outfile:
    json.dump(matched_cards, outfile)

print "blanks:"
print number_of_blanks
if (len(failed_cards) > 0):
    print "failed:"
    print failed_cards
# print matched_cards
# cv2.imshow("test", edged_screen)
# cv2.waitKey(0)
# sys.exit()
