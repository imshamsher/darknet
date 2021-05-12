import cv2


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        xmin, ymin, xmax, ymax = int(x), int(y), int(w), int(h)
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        text = detection[0].decode() + " " + str(round(detection[1] * 100, 2))
        offset = (pt1[0], pt1[1] - 5)
        color = [0, 255, 0]
        thichkness = 1
        img = draw_text_with_background(img, text,offset, color, thichkness )
        print(detection[0].decode() + " [" + str(round(detection[1] * 100, 2)) + "]")
    return img


def draw_text_with_background(img, text, offset, color, thichness):
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX

    # set the rectangle background to white
    rectangle_bgr = (70,70,70)
    # set some text
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # set the text start position
    text_offset_x = offset[0]
    text_offset_y = offset[1]
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x-2, text_offset_y+3), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=thichness)
    return img
