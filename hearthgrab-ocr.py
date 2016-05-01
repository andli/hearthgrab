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
from collections import OrderedDict

import cv2
import numpy as np
from PIL import ImageGrab

from colorlabeler import ColorLabeler


def update_progress(progress, counter):
    sys.stdout.write("\r[{0}{1}] {2}% ({3})".format('#' * (int(math.ceil(progress * PROGRESS_BAR_LENGTH))),
                                                    ' ' * (int(math.floor((1 - progress) * PROGRESS_BAR_LENGTH))),
                                                    int(round(progress * 100)),
                                                    counter))
    sys.stdout.flush()


MATCHING_METHODS = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
                    'cv2.TM_SQDIFF_NORMED']
PROGRESS_BAR_LENGTH = 20

CARD_POSITIONS = [
    [(0.028, 0.295), (0.231, 0.342)],
    [(0.266, 0.295), (0.469, 0.342)],
    [(0.504, 0.295), (0.707, 0.342)],
    [(0.740, 0.295), (0.946, 0.342)],
    [(0.028, 0.731), (0.231, 0.778)],
    [(0.266, 0.731), (0.469, 0.778)],
    [(0.504, 0.731), (0.707, 0.778)],
    [(0.740, 0.731), (0.946, 0.778)]]

PAGE_TURN_TIME = 0.3  # 0.3
CLASS_NAMES_IN_ORDER = ['druid', 'hunter', 'mage', 'paladin', 'priest', 'rogue', 'shaman', 'warlock', 'warrior',
                        'neutral']

# Determined constants
WINDOW_RECTANGLE = []
CARD_PAGE_RECTANGLE = []
NEXT_PAGE_CLICK_POINT = []

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
    # cropped_window_image_rgb = cv2.cvtColor(cropped_window_image, cv2.COLOR_BGR2RGB)
    return cropped_window_image


def find_card_page(cropped_window_image):
    page_color_hsv = [(16, 45, 140), (28, 140, 255)]
    page_hsv = cv2.cvtColor(cropped_window_image, cv2.COLOR_RGB2HSV)
    page_mask = cv2.inRange(page_hsv, page_color_hsv[0], page_color_hsv[1])

    # Find the index of the largest contour
    _, contours, _ = cv2.findContours(page_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    h, w = cropped_window_image.shape[:2]
    contours_mask = np.zeros((h, w, 3), np.uint8)
    contours_mask = cv2.cvtColor(contours_mask, cv2.COLOR_RGB2GRAY)
    cv2.drawContours(contours_mask, [cnt], 0, (255, 255, 255))

    lines = cv2.HoughLinesP(contours_mask, 12, math.pi / 2, h / 2, minLineLength=h * 0.7, maxLineGap=h * 0.7)
    # print lines

    contours_mask = cv2.cvtColor(contours_mask, cv2.COLOR_GRAY2RGB)
    x_values = []
    y_values = []
    count_lines = [0, 0]
    for line in lines:  #
        lx1, ly1, lx2, ly2 = line[0]
        if lx1 == lx2:  # vertical line
            count_lines[0] += 1
            x_values.append(lx1)
        if ly1 == ly2:  # horizontal line
            count_lines[1] += 1
            y_values.append(ly1)
        cv2.line(contours_mask, (lx1, ly1), (lx2, ly2), (0, 255, 0), 1)

    # cv2.imshow("allcards", contours_mask)
    # cv2.waitKey(0)
    # sys.exit()
    if count_lines[0] < 2 or count_lines[1] < 2:
        sys.exit("Error finding card page!")

    x, y, w, h = min(x_values), min(y_values), max(x_values) - min(x_values), max(y_values) - min(y_values)

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
    crop_x, crop_y, crop_w, crop_h = CARD_PAGE_RECTANGLE
    cropped_card_page_image = im_numpy[crop_y:crop_y + crop_h,
                              crop_x:crop_x + crop_w]  # NOTE: its img[y: y + h, x: x + w]
    crop_cards = cv2.cvtColor(cropped_card_page_image, cv2.COLOR_BGR2RGB)

    return crop_cards


# Loop all pages
print("> Looping all pages")

start_time = time.clock()
grabbed_cards = []
current_class = None
page_count = 1
card_rarities = OrderedDict({
    "legendary": 0,
    "epic": 0,
    "N/A": 0,
    "uncommon": 0,
    "common": 0})
goldens = 0
cl = ColorLabeler()


def find_minmax_hsv(golden_im):
    global h
    hmax, smax, vmax = 0, 0, 0
    hmin, smin, vmin = 255, 255, 255
    for pixelrow in golden_im:
        for pixel in pixelrow:
            b, g, r = pixel
            val = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)
            h, s, v = val[0][0]
            if h > hmax:
                hmax = h
            if s > smax:
                smax = s
            if v > vmax:
                vmax = v
            if h < hmin:
                hmin = h
            if s < smin:
                smin = s
            if v < vmin:
                vmin = v
    print str(card_position) + " max"
    print hmax, smax, vmax
    print str(card_position) + " min"
    print hmin, smin, vmin


top_fail_rarity = [0.0, 0, 0]
top_fail_x2 = [0.0, 0, 0]
top_fail_golden = [0.0, 0, 0]
while True:
    # Move the cursor so it does not trigger visual effects
    win32api.SetCursorPos((NEXT_PAGE_CLICK_POINT[0], NEXT_PAGE_CLICK_POINT[1]))

    # Take screenshot
    cards_area = screenshot_and_crop_to_card_page()
    w = CARD_PAGE_RECTANGLE[2]
    h = CARD_PAGE_RECTANGLE[3]

    card_position = 0
    for rect in CARD_POSITIONS:
        card_position += 1
        x1, y1 = rect[0]
        x2, y2 = rect[1]
        pt1 = int(round(x1 * w)), int(round(y1 * h))
        pt2 = int(round(x2 * w)), int(round(y2 * h))
        crop_half_side = int(round(h * 0.009))
        crop_rarity_x = pt1[0] + (pt2[0] - pt1[0]) / 2 - crop_half_side
        crop_rarity_y = pt2[1] - crop_half_side / 4 * 3
        crop_golden_x = pt1[0] + crop_half_side / 2
        # crop_golden_y = pt1[1] + (pt2[1] - pt2[1]) / 2 - crop_half_side - int(round((pt2[1] - pt1[1]) * 1.2))
        crop_golden_y = pt1[1] + (pt2[1] - pt2[1]) / 2 + int(round((pt2[1] - pt1[1]) * 1.0))
        crop_x2_x = pt1[0] + (pt2[0] - pt1[0]) / 2 - int(round(4.5 * crop_half_side))
        crop_x2_y = pt2[1] + (pt2[1] - pt1[1]) * 3

        # Find the ellipse
        rarity_image = cards_area[crop_rarity_y:crop_rarity_y + 2 * crop_half_side,
                       crop_rarity_x:crop_rarity_x + 2 * crop_half_side]
        rarity_hsv = cv2.cvtColor(rarity_image, cv2.COLOR_BGR2HSV)

        colors_hsv = OrderedDict({
            "legendary": [(6, 150, 50), (26, 255, 255)],  # (170, 122, 45),
            "epic": [(133, 50, 50), (153, 255, 255)],  # (136, 60, 160),
            "uncommon": [(98, 75, 50), (118, 255, 255)],  # (52, 107, 184),
            "common": [(97, 0, 50), (117, 75, 255)]})  # (127, 139, 152)
        golden_color_hsv = [(18, 120, 176), (23, 255, 255)]
        # golden_color_hsv = [(17, 120, 176), (22, 255, 255)]
        # golden_color_hsv = [(13, 150, 121), (29, 210, 255)]
        x2_color_hsv = [(18, 110, 95), (22, 165, 210)]

        # ------ DETECT GOLDENS -------
        golden_image = cards_area[crop_golden_y:crop_golden_y + 2 * crop_half_side,
                       crop_golden_x:crop_golden_x + 2 * crop_half_side]
        golden_hsv = cv2.cvtColor(golden_image, cv2.COLOR_BGR2HSV)

        golden = False
        golden_mask = cv2.inRange(golden_hsv, golden_color_hsv[0], golden_color_hsv[1])
        # cv2.imshow("mask" + str(card_position), golden_mask)
        pixel_density = float(cv2.countNonZero(golden_mask)) / float(golden_mask.size)
        if pixel_density > 0.10:
            golden = True
            goldens += 1
            # print page_count, card_position, pixel_density
            # elif pixel_density > 0.07:
            # print "fail: ", str(page_count), str(card_position), str(pixel_density)
        else:
            if pixel_density > top_fail_golden[0]:
                top_fail_golden[0] = pixel_density
                top_fail_golden[1] = page_count
                top_fail_golden[2] = card_position

        # ------ DETECT 2X -------
        x2_image = cards_area[crop_x2_y:crop_x2_y + 2 * crop_half_side,
                   crop_x2_x:crop_x2_x + 2 * crop_half_side]
        x2_hsv = cv2.cvtColor(x2_image, cv2.COLOR_BGR2HSV)

        x2 = False
        x2_mask = cv2.inRange(x2_hsv, x2_color_hsv[0], x2_color_hsv[1])
        pixel_density = float(cv2.countNonZero(x2_mask)) / float(x2_mask.size)
        # cv2.imshow("hsv" + str(count), x2_hsv)
        # cv2.imshow("mask" + str(count), x2_mask)
        # print pixel_density
        if pixel_density > 0.4:
            x2 = True
        else:
            if pixel_density > top_fail_x2[0]:
                top_fail_x2[0] = pixel_density
                top_fail_x2[1] = page_count
                top_fail_x2[2] = card_position

        # ------ DETECT RARITY -------
        # cv2.imshow("hsv" + str(count), mask)
        color = 'N/A'
        pixel_density = 0.0
        for (i, (name, hsv)) in enumerate(colors_hsv.items()):
            rarity_mask = cv2.inRange(rarity_hsv, hsv[0], hsv[1])
            pixel_density_new = float(cv2.countNonZero(rarity_mask)) / float(rarity_mask.size)
            if pixel_density_new > 0.40:
                if pixel_density_new > pixel_density:
                    pixel_density = pixel_density_new
                    color = name
            else:
                if pixel_density_new > top_fail_rarity[0]:
                    top_fail_rarity[0] = pixel_density_new
                    top_fail_rarity[1] = page_count
                    top_fail_rarity[2] = card_position

        card_rarities[color] += 1
        #        cv2.imshow("original", rarity_image)
        #        cv2.waitKey(0)
        #        sys.exit()

        # Card text
        cv2.rectangle(cards_area, (crop_rarity_x, crop_rarity_y),
                      (crop_rarity_x + 2 * crop_half_side, crop_rarity_y + 2 * crop_half_side), (0, 255, 0))
        cv2.rectangle(cards_area, (crop_golden_x, crop_golden_y),
                      (crop_golden_x + 2 * crop_half_side, crop_golden_y + 2 * crop_half_side), (0, 255, 0))
        cv2.rectangle(cards_area, (crop_x2_x, crop_x2_y),
                      (crop_x2_x + 2 * crop_half_side, crop_x2_y + 2 * crop_half_side), (0, 255, 0))
        cv2.putText(cards_area, color, (crop_rarity_x - 50, crop_rarity_y - 190), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 0), 2)
        if golden:
            cv2.putText(cards_area, "golden", (crop_rarity_x - 50, crop_rarity_y - 175), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 0), 2)
        if x2:
            cv2.putText(cards_area, "2X", (crop_rarity_x - 50, crop_rarity_y - 160), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 0), 2)
            # Rarity gem
            # cv2.rectangle(cards_area, pt1, pt2, (255, 0, 255), 1)

    # cv2.imshow(str(page_count) + "allcards", cards_area)
    # cv2.waitKey(0)
    # sys.exit()
    #
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

    # print page_count
    #
    # #
    #
    # last_window_page_no = cards_area[PAGE_NO_POSITION[1]: PAGE_NO_POSITION[1] + PAGE_NO_SIZE[1],
    #                       PAGE_NO_POSITION[0]: PAGE_NO_POSITION[0] + PAGE_NO_SIZE[0]]
    # next_window_page_no = new_window[PAGE_NO_POSITION[1]: PAGE_NO_POSITION[1] + PAGE_NO_SIZE[1],
    #                       PAGE_NO_POSITION[0]: PAGE_NO_POSITION[0] + PAGE_NO_SIZE[0]]
    # page_no_result = cv2.matchTemplate(last_window_page_no, next_window_page_no, eval(MATCHING_METHODS[1]))
    # (_, max_page_matching_value, _, _) = cv2.minMaxLoc(page_no_result)
    # if (max_page_matching_value > 0.999):
    if page_count == 95:
        print card_rarities
        print "Goldens: " + str(goldens)
        print "Top fail golden (0.1): " + str(top_fail_golden)
        print "Top fail x2 (0.40): " + str(top_fail_x2)
        print "Top fail rarity (0.40): " + str(top_fail_rarity)
        # cv2.waitKey(0)
        # sys.exit()
        break
    else:
        # Click to the next page
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_ABSOLUTE, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP | win32con.MOUSEEVENTF_ABSOLUTE, 0, 0)
        # win32api.ClipCursor((0,0,0,0))
        time.sleep(PAGE_TURN_TIME);
        page_count += 1
        # Check if we are on the last page
        new_window = screenshot_and_crop_to_card_page()

end_time = time.clock()
print("Collected all cards in %.2f seconds." % (end_time - start_time))
print("Scanned " + str(page_count) + " pages.")

sys.exit()
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

# print "blanks:"
# print number_of_blanks
if (len(failed_cards) > 0):
    print "failed:"
    print failed_cards
# print matched_cards
# cv2.imshow("test", edged_screen)
# cv2.waitKey(0)
# sys.exit()
