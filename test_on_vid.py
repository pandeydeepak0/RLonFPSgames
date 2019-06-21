import cv2
import numpy as np 

def draw_lanes(img, lines, color=[0, 255, 255], thickness=0.2):

    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker 
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []  
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]

                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
    except Exception as e:
        print(str(e))
        
def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    #romove color overlapping
    if len(img.shape) > 2:
        channel_count = img.shape[2] 
        color_change = (255,) * channel_count
    else:
        color_change = 255
    # fill the mask
    cv2.fillPoly(mask, vertices, color_change)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

video= 'Data/gamevid.mp4'
video_cap= cv2.VideoCapture(video)
car_cascade= cv2.CascadeClassifier('classifiers/cars.xml')

while 1:
    #capture video frames
    ret, sct_img = video_cap.read()
    #convert color channel from BGR to GRAY
    GRAYimg= cv2.cvtColor(sct_img, cv2.COLOR_BGR2GRAY)
    #convert color channel from BGR to HSL(Hue, Saturation and Color)
    HSLimg= cv2.cvtColor(sct_img, cv2.COLOR_BGR2HLS)
    #find the Lchannel
    Lchannel = HSLimg[:,:,1]

    #detect cars using cascade classifier
    cars = car_cascade.detectMultiScale(GRAYimg, 1.1, 1)
    for (x,y,w,h) in cars:
        cv2.rectangle(sct_img,(x,y),(x+w,y+h),(0,0,255),2)      

    #find the range for white color
    low_white = np.array([220, 220, 220])
    high_white = np.array([255, 255, 255])

    #find the range for yellow color
    low_yellow = np.array([150, 150, 0])
    high_yellow = np.array([255, 255, 100])

    #mask the ranges in the screen
    white_mask = cv2.inRange(Lchannel, 210, 255)
    yellow_mask = cv2.inRange(Lchannel, 140, 210)

    #perform bitwise and ops to highlighten white portion of screen
    yellow_res= cv2.bitwise_and(sct_img, sct_img, mask= yellow_mask)
    white_res= cv2.bitwise_and(sct_img, sct_img, mask= white_mask)
    res= cv2.bitwise_or(yellow_res, white_res, mask= None)

    #find the vertices enclosing the region
    region= np.array([[10, 440], [10, 460], [260, 170], [360, 170], [670, 460], [670, 440],], np.int32)
    
    #perform canny edge detection to find high contrast points
    processed_img = cv2.Canny(GRAYimg, threshold1=250, threshold2=350)
    #perform gaussian blur to smooth image
    processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0 )

    #mask the vertices polygon with other screen
    processed_img_region=roi(processed_img, [region])
    res_region= roi(res, [region])
    sct_img_region= roi(sct_img, [region])

    #draw hough lines to find strongest lines
    lines = cv2.HoughLinesP(processed_img_region, 1, np.pi/180, 180, 20, 15)
    try:
        l1, l2 = draw_lanes(res,lines)
        cv2.line(res, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(res, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(res, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                
                
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass
    
    cv2.imshow('window1', sct_img)
    cv2.imshow('screen', res)
    cv2.imshow('window', processed_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break