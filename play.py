import numpy as np
import cv2
from mss.linux import MSS as mss
from mss.tools import to_png
from PIL import Image, ImageDraw, ImageChops
import time
import pyautogui as pg
import pytesseract
import os
import random

def Init():
    global acp_short, acp_long, debug, wc_time, scr_resolution
    slow_machine = False
    #slow_machine = True

    if slow_machine:
        acp_short = (0.6, 0.9) # after click pause
        acp_long = (3, 5) # after click pause
        scr_resolution = {'x': 1280, 'y': 1024}
    else:
        acp_short = (0.45, 0.7) # after click pause
        acp_long = (2, 4) # after click pause
        scr_resolution = {'x': 1920, 'y': 1080}
    debug = False
    wc_time = 3000
    random.seed(version=2)


def ClickRandDelay():
    click_time = random.uniform(0.1, 0.5)
    pg.mouseDown()
    time.sleep(click_time)
    pg.mouseUp()


def MoveCursorRandPath(dest_x, dest_y, dur = 0.7):
    ## Any duration less than this is rounded to 0.0 to instantly move the mouse.
    pg.MINIMUM_DURATION = 0  # Default: 0.1
    ## Minimal number of seconds to sleep between mouse moves.
    pg.MINIMUM_SLEEP = 0  # Default: 0.05
    ## The number of seconds to pause after EVERY public function call.
    pg.PAUSE = 0  # Default: 0.1

    random.seed(version=2)
    x1, y1 = pg.position()  # Starting position
    x2, y2 = dest_x, dest_y
    x_diff = x2 - x1
    y_diff = y2 - y1
    if ((x_diff == 0) and (y_diff == 0)):
        return 0

    x = x1
    y = y1
    x_min_step, x_max_step = 4, 30 # min and max step length for randomizer
    y_min_step, y_max_step = 4, 30

    # Задаем коэффиценты, определяющие направление движения
    kx = 1 if (x2 >= x) else -1
    ky = 1 if (y2 >= y) else -1

    # Подгоняем количество точек для координат с наименьшим перемещением,
    # чтобы по обеим осям курсор  прибыл в пункт назначения примерно синхронно
    if (abs(x_diff) > abs(y_diff)):
        x_avg_step = (x_min_step + x_max_step)/2
        x_steps = round(x_diff/x_avg_step) if (abs(round(x_diff/x_avg_step)) > 0) else 1
        y_avg_step = round(y_diff/x_steps)
        y_min_step, y_max_step = abs(round((y_avg_step/3)*2)), abs(round((y_avg_step/3)*5))
        if ((y_min_step <= 1) or (y_max_step <= 1)):
            y_min_step, y_max_step = 2, 8
    else:
        y_avg_step = (y_min_step + y_max_step)/2
        y_steps = round(y_diff/y_avg_step) if (abs(round(y_diff/y_avg_step)) > 0) else 1
        x_avg_step = round(x_diff/y_steps)
        x_min_step, x_max_step = abs(round((x_avg_step/3)*2)), abs(round((x_avg_step/3)*5))
        if ((x_min_step == 0) or (x_max_step ==0)):
            x_min_step, x_max_step = 2, 8
    if (debug):
        print(x_min_step, x_max_step)
        print(y_min_step, y_max_step)
    points = []
    x_nearly, y_nearly = False, False
    # Пока не достигли координат назначения, прибавляем к каждой координате
    # случайное значение из заданного для координаты интервала.
    # При этом направление движения задается коэффицентом, умножамым на прибавляемое значение
    while ((x != x2) or (y != y2)):
        step_x = random.randint(x_min_step, x_max_step)
        step_y = random.randint(y_min_step, y_max_step)
        x += kx*step_x
        y += ky*step_y

        x_ready_for_finish = (kx*x >= kx*x2) and (ky*y + ky*y_max_step >= ky*y2)
        y_ready_for_finish = (ky*y >= ky*y2) and (kx*x + kx*x_max_step >= kx*x2)
        x_ready_y_not = (kx*x > kx*x2 + kx*x_max_step) and (ky*y + ky*y_max_step < ky*y2)
        y_ready_x_not = (ky*y > ky*y2 + ky*y_max_step) and (kx*x + kx*x_max_step < kx*x2)

        if x_ready_for_finish or y_ready_for_finish or (x_nearly and y_nearly):
            x = x2
            y = y2
        elif x_ready_y_not:
            x_nearly = True
            kr = random.uniform(0.9, 1.3)
            x = round(x - kx*step_x*kr)
        elif y_ready_x_not:
            y_nearly = True
            kr = random.uniform(0.9, 1.3)
            y = round(y - ky*step_y*kr)
        if (debug):
            print (x, y)
        points.append((x,y))

    #Слегка меняем продолжительность движения
    duration = dur
    d_p_rand = random.uniform(0.01, 0.2)
    d_n_rand = random.uniform(-0.2, -0.01)
    d_rand = d_p_rand + d_n_rand
    timeout = (duration + d_rand) / len(points)

    for pt in points:
        pg.moveTo(pt[0], pt[1])
        t_rand = random.uniform(0.0001, 0.001)
        time.sleep(timeout+t_rand)


#def RandomNearbyPoint(x, y, x_slip = 5, y_slip = 5):
#    tm_default = 0.5
#    params = {'x': x, 'y': y, 'tm': 0.5}
#    min_x, max_x = x-x_slip, x+x_slip
#    min_y, max_y = y-y_slip, y+y_slip
#    params['x'] = random.randint(min_x, max_x)
#    params['y'] = random.randint(min_y, max_y)
#    params['tm'] = random.uniform(0.67, 0.9)
#    return params


def MoveCurAndClick(x = 0, y = 0,
                    only_move = False,
                    #rand_mov_chance = 0.001,
                    x_slip = 5, y_slip = 5):
    print('x='+str(x)+', y='+str(y)) if (debug) else 0
    mv = {'x': x, 'y': y, 'tm': 0.5}
    min_x, max_x = x-x_slip, x+x_slip
    min_y, max_y = y-y_slip, y+y_slip
    mv['x'] = random.randint(min_x, max_x)
    mv['y'] = random.randint(min_y, max_y)
    mv['tm'] = random.uniform(0.67, 0.9)
    print(mv)

    MoveCursorRandPath(mv['x'], mv['y'], mv['tm'])
    SleepRand(acp_short[0], acp_short[1])
    if (only_move == False):
        ClickRandDelay()
        SleepRand(acp_short[0], acp_short[1])
        #RandomMouseMove(rand_mov_chance)
        #SleepRand(acp_short[0], acp_short[1])


def SleepRand(sleep_min=3, sleep_max=10):
    time.sleep(random.uniform(sleep_min, sleep_max))


def MultiClick(count=1):
    clicked = 0
    while (clicked < count):
        after_click_time = random.uniform(0.45, 0.9)
        ClickRandDelay()
        time.sleep(after_click_time)
        # немного смещаем курсор, если время задержки между кликами превысило определенный порог
        if (after_click_time > 0.75):
            x, y = pg.position()
            print('MultiClick. pg.position()=',pg.position())
            MoveCurAndClick(x, y, only_move=True)
        clicked += 1

# Функция для имитации человеческого присутсвия
# Двигает мышь в произвольное место в рамках заданного разрешения экрана
# Движение может быть выполнено от 0 до 2 раз.
# Количество определяется случайным образом в зависимости от указанной вероятности.
# По умолчанию вероятность 40%. Вероятность второй итерации уменьшается в 2 раза.
def RandomMouseMove(treshold = 0.4):
    i = 0
    while (i != 2):
        i += 1
        if (random.random() <= (treshold/i)):
            rand_x = random.randint(49, scr_resolution['x']-69)
            rand_y = random.randint(53, scr_resolution['y']-65)
            MoveCurAndClick(rand_x, rand_y, only_move = True)


def ScrollMouse(number = 10, direction = 'DOWN'):
    scrolled = 0
    interim = 0
    s_step = 1 if (direction == 'UP') else -1
    min_scroll, max_scroll = number - 2, number + 3
    target = random.randint(min_scroll, max_scroll)

    interval = random.randint(5, 7)
    while (scrolled < target):
        interim = scrolled + interval
        while ((scrolled < interim) and (scrolled < target)):
            pg.scroll(s_step)
            scrolled += 1
            print(scrolled)
            SleepRand(0.1, 0.2)
        SleepRand(0.6, 0.9)


def DragAndDrop(start_point, end_point):
    #start_point = {'x': 2, 'y': 4}
    #end_point = {'x': 12, 'y': 4}
    MoveCurAndClick(start_point['x'], start_point['y'], only_move=True)
    pg.mouseDown()
    MoveCurAndClick(end_point['x'], end_point['y'], only_move=True)
    pg.mouseUp()


# Convert cyrilic symbols to latin layout
def ConvertToLat(text):
    text2 = ''
    text = text.lower()
    map = {'й': 'q', 'ц': 'w', 'у': 'e', 'к': 'r', 'е': 't', 'н': 'y',
           'г': 'u', 'ш': 'i', 'щ': 'o', 'з': 'p', 'х': '[', 'ъ': ']',
           'ф': 'a', 'ы': 's', 'в': 'd', 'а': 'f', 'п': 'g', 'р': 'h',
           'о': 'j', 'л': 'k', 'д': 'l', 'ж': ';', 'э': '\'', 'я': 'z',
           'ч': 'x', 'с': 'c', 'м': 'v', 'и': 'b', 'т': 'n', 'ь': 'm',
           'б': ',', 'ю': '.'
    }
    for t in text:
        text2 += map[t] if (t in map) else t
    return text2

def ValidParam(valid_list, verifiable):
    if (verifiable in valid_list):
        return True
    else:
        #print('"' + s + '"' + ' is invalid parameter')
        print('"%s" is invalid parameter' %verifiable)
        print('Valid parameters are:', end = ' ')
        for l in valid_list:
            if (l != valid_list[len(valid_list)-1]):
                print(l, end = ', ')
            else:
                print(l)
        return False

class Layout(object):
    def __init__(self):
        self.cur_layout = ''
        self.valid_layouts = ('EN', 'RU')
        self.screenshot_area = {'top': 0, 'left': 0, 'width': scr_resolution['x'], 'height': scr_resolution['y']}
        self.layout_templ = {'en': './templates/icon_layout_en_linux2.png',
                             'ru': './templates/icon_layout_ru_linux2.png'}

    def Check(self):
        layout_icon = Icon(shot_area = self.screenshot_area,
                           icon_template = self.layout_templ['en'],
                           icon_treshold = 0.99)
        layout_icon.icon_template = self.layout_templ['en']
        en_icon_exist = layout_icon.Find()
        layout_icon.icon_template = self.layout_templ['ru']
        ru_icon_exist = layout_icon.Find()
        if (en_icon_exist):
            self.cur_layout = 'EN'
        elif (ru_icon_exist):
            self.cur_layout = 'RU'
        return self.cur_layout

    def Change(self, need_layout):
        need_layout = need_layout.upper()
        print('Layout.Change')
        print('need_layout =', need_layout)
        print('self.cur_layout =',self.cur_layout)

        if not ValidParam(self.valid_layouts, need_layout):
            return 1

        if (need_layout.upper() != self.cur_layout):
            print('press shift+ctrl')
            pg.hotkey('shift', 'ctrl')
            SleepRand(acp_long[0], acp_long[1]) # delay for changing layout

        return self.Check()


def EnterText(text='coolrock\4'):
    text = text.lower()
    cyr_sym = ('й', 'ц', 'у', 'к', 'е', 'н', 'г', 'ш', 'щ', 'з', 'х',
               'ъ', 'ф', 'ы', 'в', 'а', 'п', 'р', 'о', 'л', 'д', 'ж',
               'э', 'я', 'ч', 'с', 'м', 'и', 'т', 'ь', 'б', 'ю', 'ё')
    lat_sym = ('q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o',
               'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',
               'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm')
    layout = Layout()
    cur_layout = layout.Check()

    for w in text:
        if (w in cyr_sym):
            w = ConvertToLat(w)
            if (cur_layout == 'EN'):
                cur_layout = layout.Change('RU')
        elif ((w in lat_sym) and (cur_layout == 'RU')):
            cur_layout = layout.Change('EN')

        SleepRand(0.05, 0.1)
        pg.write(w, random.uniform(0.1, 0.35))
    SleepRand(acp_long[0], acp_long[1])


def CutThePict(area, png=False):
    area = area
    sct = mss()
    img = sct.grab(area)
    if (png == False):
        img_np = np.array(img)

        #img_name = str(time.time())[:10]+str(time.time())[-2:]
        #output = "scr/"+img_name+".png"
        #to_png(img.rgb, img.size, output=output)

        return img_np
    else:
        img_name = str(time.time())[:10]+str(time.time())[-2:]
        output = "scr/"+img_name+".png"
        to_png(img.rgb, img.size, output=output)
        return output


def FindObject(full_img, templ_img, treshold):
    position = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
    found = False   # флаг наличия изображения
    gray_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)   # преобразуем её в серую
    template = cv2.imread(templ_img, cv2.IMREAD_GRAYSCALE)  # искомый объект преобразуем в серый

    #cv2.imshow("img", template)
    #cv2.waitKey(3000)
    #cv2.destroyAllWindows()

    h, w = template.shape
    half_h, half_w = round(h/2), round(w/2)

    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED) # ищем
    top_left = np.where(result >= treshold)[::-1]   # координаты сразу преводим в формат x:y
    if (len(top_left[0]) > 0):
        top_left = (top_left[0][0], top_left[1][0]) # избавляемся от лишнего обертывания
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center_point = (top_left[0] + half_w, top_left[1] + half_h)

        if (debug):
            cv2.rectangle(full_img, top_left, bottom_right, (0, 255, 0), 1)
            cv2.circle(full_img, center_point, 1, (0, 255, 0), 2)

            img_name = str(time.time())[:10]+str(time.time())[-2:]
            path = 'scr'
            output = "rect_"+img_name+".jpg"
            cv2.imwrite(os.path.join(path , output), full_img)

            #cv2.imshow("img", full_img)
            #cv2.waitKey(15000)
            #cv2.destroyAllWindows()

        position = {'top_left': top_left, 'bottom_right': bottom_right, 'center_point': center_point}
        found = True
    return found, position


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(wc_time)
    cv2.destroyAllWindows()


def grayscale_3_levels(gray_img):
    high = 255
    while(1):
        low = high - 85
        col_to_be_changed_low = np.array([low])
        col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(gray_img, col_to_be_changed_low, col_to_be_changed_high)
        if (high > 85):
            gray_img[curr_mask > 0] = (255)
        else:
            gray_img[curr_mask > 0] = (0)
        high -= 85
        if(low == 0):
            break
    return gray_img


def ImgToBW(gray_img):
    high = 255
    while(1):
        low = high - 15
        col_to_be_changed_low = np.array([low])
        col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(gray_img, col_to_be_changed_low, col_to_be_changed_high)
        if (high >= 140):
            gray_img[curr_mask > 0] = (255)
        else:
            gray_img[curr_mask > 0] = (0)
        high -= 15
        if(low == 0):
            break
    return gray_img


def grayscale_17_levels(gray_img):
    high = 255
    while(1):
        low = high - 15
        col_to_be_changed_low = np.array([low])
        col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(gray_img, col_to_be_changed_low, col_to_be_changed_high)
        gray_img[curr_mask > 0] = (high)
        high -= 15
        if(low == 0):
            break
    return gray_img

def recognize_image(rec_area):
    img = CutThePict(rec_area)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = ImgToBW(img)
    if (debug):
        cv2.imshow('image', img)
        cv2.waitKey(wc_time)
        cv2.destroyAllWindows()
    #text='test'
    text = pytesseract.image_to_string(img, lang="rus")
    print(text)
    return text

def ColorExist(icon_area, color_range):
    low_col = color_range['low_col']
    high_col = color_range['high_col']
    img = CutThePict(icon_area)
    #img = cv2.imread('daily_icon.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if (debug):
        viewImage(img)
    print(type(img), img.shape)
    print(low_col, high_col)

    col_range = cv2.inRange(img, low_col, high_col)

    if (debug):
        cv2.imshow('col_range', col_range)
        cv2.waitKey(wc_time)
    if (len(img[col_range > 0]) > 0):
        print(img[col_range > 0])
        return True
    else:
        return False


def MoveCheck(check_area):
    img_name1 = CutThePict(check_area, png=True)
    time.sleep(1.3)
    img_name2 = CutThePict(check_area, png=True)
    img1 = Image.open(img_name1)
    img2 = Image.open(img_name2)
    os.remove(img_name1)
    os.remove(img_name2)

    diff = ImageChops.difference(img1, img2)
    # показываем разницу
    #diff.show()
    #print(diff.getbbox())
    if (diff.getbbox() != None):
        return True
    else:
        return False


class Window(object):

    def __init__(self,
                 shot_area,
                 h_substr = 'no_header',
                 h_template = 'no_header',
                 h_treshold = 0,
                 b_template = 'no_button',
                 b_treshold = 0.7,
                 e_template = 'no_element',
                 e_treshold = 0.8):
        self.header_substr = h_substr
        self.shot_area = shot_area
        self.header_template = h_template
        self.header_treshold = h_treshold
        self.element_template = e_template
        self.element_treshold = e_treshold
        self.action_button_template = b_template
        self.action_button_treshold = b_treshold
        self.close_button_template = './templates/b_close.png'
        self.close_button_treshold = 0.99
        self.screen = CutThePict(self.shot_area)
        self.action_button_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
        self.close_button_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}

    def FreshScreenShot(self):
        self.screen = CutThePict(self.shot_area)
        SleepRand(0.1, 0.3)

    def CheckHeader(self):
        if (self.header_substr == 'no_header'):
            return False

        header_exist, header_border_pos = FindObject(self.screen, self.header_template, self.header_treshold)
        n_top = header_border_pos['top_left'][1]
        n_right = header_border_pos['bottom_right'][0]
        n_height = (header_border_pos['bottom_right'][1])-(header_border_pos['top_left'][1])
        n_width = 500
        if (header_exist):
            #time.sleep(20)
            header_area = {'top': n_top, 'left': n_right, 'width': n_width, 'height': n_height}
            rec_text = recognize_image(header_area)
            pos = rec_text.upper().find(self.header_substr.upper())
            print(self.header_substr)
            print('pos = '+str(pos))
            if (pos > -1):
                print('Слово "'+self.header_substr+'" найдено')
                return True
            else:
                return False
                print('Слово "'+self.header_substr+'" не найдено')
        else:
            return False

    def CheckElement(self):
        element_exist, element_pos = FindObject(self.screen, self.element_template, self.element_treshold)
        return element_exist

    def FindButton(self, type,
                   move_to = False,
                   press = False,
                   delay = True,
                   rand_mov_chance = 0.3):
        if (type.upper() == 'ACTION'):
            if (debug):
                print(self.action_button_template)
                print(self.action_button_treshold)
            button_temlate = self.action_button_template
            button_treshold = self.action_button_treshold
            if (button_temlate.upper() == 'NO_BUTTON'):
                return False
        elif (type.upper() == 'CLOSE'):
            button_temlate = self.close_button_template
            button_treshold = self.close_button_treshold

        button_exist, button_pos = FindObject(self.screen, button_temlate, button_treshold)

        if (type.upper() == 'ACTION'):
            self.action_button_pos = button_pos
        elif (type.upper() == 'CLOSE'):
            self.close_button_pos = button_pos

        if (button_exist and move_to):
            MoveCurAndClick(button_pos['center_point'][0], button_pos['center_point'][1], only_move = True)

        if (button_exist and press):
            MoveCurAndClick(button_pos['center_point'][0], button_pos['center_point'][1])
            RandomMouseMove(rand_mov_chance)
            if (delay == True):
                SleepRand(acp_long[0], acp_long[1])

        return button_exist

    def PressButton(self, type, delay = True, rand_mov_chance = 0.3):
        if (type.upper() == 'ACTION'):
            button_pos = self.action_button_pos
        elif (type.upper() == 'CLOSE'):
            button_pos = self.close_button_pos

        MoveCurAndClick(button_pos['center_point'][0], button_pos['center_point'][1])
        RandomMouseMove(rand_mov_chance)
        if (delay == True):
            SleepRand(acp_long[0], acp_long[1])


class Icon(object):
#shot_area, ico_template, b_template, h_treshold,
    def __init__(self, shot_area, icon_template, icon_treshold = 0.7):
        """Constructor"""
        self.shot_area = shot_area
        self.icon_template = icon_template
        self.icon_treshold = icon_treshold
        self.icon_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
        #x_slip, y_slip = 0, 0

    def Find(self):
        self.screen = CutThePict(self.shot_area)
        icon_exist, self.icon_pos = FindObject(self.screen, self.icon_template, self.icon_treshold)
        #icon_exist = ColorExist(self.icon_area, self.icon_color_range)
        return icon_exist

    def CheckNumber(self, number_color_range):
        #number_color_range = {'low_col': (18,11,85), 'high_col': (38,25,151)} (BGR)
        self.screen = CutThePict(self.shot_area)
        icon_exist, self.icon_pos = FindObject(self.screen, self.icon_template, self.icon_treshold)
        # расширяем область поиска (при этом остаемся в границах разрешения экрана)
        corr = 40
        top_ext = self.icon_pos['top_left'][1]-corr
        left_ext = self.icon_pos['top_left'][0]-corr
        height_ext = (self.icon_pos['bottom_right'][1])-(self.icon_pos['top_left'][1]) + corr*2
        width_ext = (self.icon_pos['bottom_right'][0])-(self.icon_pos['top_left'][0]) + corr*2
        area_top = top_ext if (top_ext >= corr) else 0
        area_left = left_ext if (left_ext >= corr) else 0
        area_height = height_ext if (area_top + height_ext <= scr_resolution['y']) else scr_resolution['y'] - area_top
        area_width = width_ext if (area_left + width_ext <= scr_resolution['x']) else scr_resolution['x'] - area_left
        icon_area = {'top': area_top, 'left': area_left, 'width': area_width, 'height': area_height}
        icon_numbered = ColorExist(icon_area, number_color_range)
        return icon_numbered

    def Movement(self):
        i_top = self.icon_pos['top_left'][1]
        i_left = self.icon_pos['top_left'][0]
        i_width = self.icon_pos['bottom_right'][0]-self.icon_pos['top_left'][0]
        i_height = self.icon_pos['bottom_right'][1]-self.icon_pos['top_left'][1]
        icon_area = {'top': i_top, 'left': i_left, 'width': i_width, 'height': i_height}
        move = MoveCheck(icon_area)
        return move

    def Click(self, delay = True, rand_mov_chance = 0.3):
        MoveCurAndClick(self.icon_pos['center_point'][0], self.icon_pos['center_point'][1])
        RandomMouseMove(rand_mov_chance)
        if (delay == True):
            SleepRand(acp_long[0], acp_long[1])


class WindowIcon(object):
    def __init__(self, shot_area, icon_template, icon_treshold = 0.7):
        self.shot_area = shot_area
        self.icon_template = icon_template
        self.icon_treshold = icon_treshold
        self.screen = CutThePict(self.shot_area)
        self.icon_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}

        self.right_border_templ = ''
        self.action_elem_templ = ''


class Vik_akk():
    login = ''
    def __init__(self):
        self.screenshot_area = {'top': 0, 'left': 0, 'width': scr_resolution['x'], 'height': scr_resolution['y']}
        self.red_number_color_range = {'low_col': (18,11,85), 'high_col': (38,25,151)}

    def PromoClose(self):
        print('PromoClose start')
        #header_substr = 'предложение'
        #e_template = './templates/header_border_1.png'
        #e_treshold = 0.5

        promo_window = Window(self.screenshot_area,
                              e_template = './templates/header_border_1.png',
                              e_treshold = 0.5)
        window_exist = promo_window.CheckElement()
        if (window_exist):
            button_exist = promo_window.FindButton('CLOSE')
            if (button_exist):
                promo_window.PressButton('CLOSE')
                ScrollMouse(number = 11, direction = 'DOWN')
                print('button_exist =', button_exist)
            else:
                print('Кнопка не обнаружена')


    def TakeDailyLoyalBonus(self):
        print('TakeDailyLoyalBonus start')
        icon_template = './templates/icon_daily_loyal.png'
        icon_treshold = 0.5
        dl_icon = Icon(self.screenshot_area, icon_template, icon_treshold)
        icon_exist = dl_icon.Find()
        if (icon_exist):
            print('Icon exist')
            dl_icon.Click()

            dl_window = Window(self.screenshot_area,
                               h_substr = 'лояльно',
                               h_template = './templates/header_border_1.png',
                               h_treshold = 0.5,
                               b_template = './templates/b_take.png',
                               b_treshold = 0.99)
            header_exist = dl_window.CheckHeader()
            if (header_exist):
                button_exist = dl_window.FindButton('ACTION', press = True)
            else:
                print('Необходимое окно не открылось')
                # добавить закрытие окна
        else:
            print('No icon')

# вынести повторяющийся код в виде атрибутов
# сделать универсальный метод для статичных иконок/окон, если это будет целесообразно (например дневная лояльность и помощь)
# разделить кнопки на "забрать" и "закрыть"
    def PressHelp(self):
        print('PressHelp start')
        icon_template = './templates/icon_clan_help.png'
        icon_treshold = 0.7
        hlp_icon = Icon(self.screenshot_area, icon_template, icon_treshold)
        icon_exist = hlp_icon.Find()
        if (icon_exist):
            print('Icon exist')
            hlp_icon.Click()

            hlp_window = Window(self.screenshot_area,
                                h_substr = 'участник',
                                h_template = './templates/header_border_1.png',
                                h_treshold = 0.5,
                                b_template = './templates/b_help.png',
                                b_treshold = 0.99)
            header_exist = hlp_window.CheckHeader()
            if (header_exist):
                button_exist = hlp_window.FindButton('ACTION', press = True)
                close_button_exist = hlp_window.FindButton('CLOSE', press = True)
        else:
            print('No icon')


    def TakeEveryDayBank(self):
        print('TakeEveryDayBank start')
        icon_template = './templates/icon_bank_part.png'
        icon_treshold = 0.9
        icon_number_cr = self.red_number_color_range
        bank_icon = Icon(self.screenshot_area, icon_template, icon_treshold)
        icon_numbered = bank_icon.CheckNumber(icon_number_cr)
        if (icon_numbered):
            print('Icon numbered')
            bank_icon.Click()

            bank_window = Window(self.screenshot_area,
                                 h_substr = 'Банк',
                                 h_template = './templates/header_border_1.png',
                                 h_treshold = 0.5,
                                 b_template = './templates/tab_bank_subscribe.png',
                                 b_treshold = 0.99)
            header_exist = bank_window.CheckHeader()
            #time.sleep(20)
            button_exist = bank_window.FindButton('ACTION')
            print('header_exist =', header_exist)
            print('button_exist =', button_exist)
            if (header_exist and button_exist):
                bank_window.PressButton('ACTION')
                print('Subscribe tab pressed')
                bank_window.action_button_template = './templates/b_take_3_bank1.png'
                bank_window.action_button_treshold = 0.8
                bank_window.FreshScreenShot()
                button_exist = bank_window.FindButton('ACTION')
                if (button_exist):
                    bank_window.PressButton('ACTION')
                    bank_window.FreshScreenShot()
                    bank_window.action_button_template = './templates/b_take_4_bank2.png'
                    button_exist = bank_window.FindButton('ACTION', press = True)
                else:
                    print('Кнопка "забрать" не обнаружена')
                close_button_exist = bank_window.FindButton('CLOSE', press = True)
            else:
                print('Заголовок не соответсвует или вкладка не обнаружена')
                close_button_exist = bank_window.FindButton('CLOSE', press = True)
        else:
            print('Icon NOT numbered')


    def TakeChestOfLoki(self):
        print('TakeChestOfLoki start')
        icon_template = './templates/icon_loki_chest.png'
        icon_treshold = 0.7
        chol_icon = Icon(self.screenshot_area, icon_template, icon_treshold)
        icon_exist = chol_icon.Find()
        if (icon_exist):
            print('Icon exist')
            moving = chol_icon.Movement()
            if (moving):
                chol_icon.Click()
                chol_window = Window(self.screenshot_area,
                                     h_substr = 'оки',
                                     h_template = './templates/header_border_loki.png',
                                     h_treshold = 0.8,
                                     b_template = './templates/b_take_loki.png',
                                     b_treshold = 0.5)
                button_exist = chol_window.FindButton('ACTION', press = True)
            else:
                print('No movement')


# Проверяем наличие новых заданий (число красном кружочке) и жмем на иконку при наличии
# Так же жмем, если необходимо выполнить задания многократно (применяя обновление заданий)
# Проверяем наличие кнопки "Забрать все" и жмем ее при наличии
# Если этой кнопки нет, ищем кнопку начать, перемещаемся на нее и прокликиваем.
# Так для каждой вкладки
# Может быть запрошено выполнение нескольких циклов заданий для какой-то из вкладок
# take_tab - порядковый номер вкладки, для которой запрошено несколько заданий
# take_number - требуемое количество циклов выполнения заданий для запрошенной вкладки
    def TakeTasks(self, take_tab = 1, take_number = 1):
        print('TakeTasks start')
        icon_template = './templates/icon_tasks.png'
        icon_treshold = 0.7
        icon_number_cr = self.red_number_color_range
        task_icon = Icon(self.screenshot_area, icon_template, icon_treshold)
        icon_numbered = task_icon.CheckNumber(icon_number_cr)
        if (icon_numbered or take_number > 1):
            print('Icon numbered')
            task_icon.Click()
            # если активных заданий нет, то увеличиваем количество циклов на 1
            # дабы выполнить ровно столько циклов заданий, сколько запрошено
            if (icon_numbered == False):
                take_number += 1
            current_tab = 0
            tabs = ('./templates/tab_tasks_pers.png',
                    './templates/tab_tasks_clan.png',
                    './templates/tab_tasks_vip.png')
            t = {'take_all': './templates/b_tasks_take_all.png',
                 'take': './templates/b_tasks_take.png',
                 'start': './templates/b_tasks_start.png',
                 'apply': './templates/b_tasks_apply.png',
                 'collapsed_elem': './templates/elem_task.png',
                 'expanded_elem': './templates/elem_task_expanded.png'}

            task_window = Window(self.screenshot_area,
                                 b_template = t['take_all'],
                                 b_treshold = 0.7,
                                 e_template = t['collapsed_elem'])

            for tab in tabs:
                current_tab += 1
                current_number = 0
                number = take_number if (current_tab == take_tab) else 1
                if (tab != tabs[0]):
                    task_window.action_button_template = tab
                    task_window.FreshScreenShot()
                    tab_exist = task_window.FindButton('ACTION')
                    if (tab_exist):
                        task_window.PressButton('ACTION')
                    else:
                        task_window.FindButton('CLOSE', press = True)
                        return 1

                while (current_number < number):
                    current_number += 1
                    task_window.action_button_template = t['take_all']
                    task_window.action_button_treshold = 0.99
                    task_window.FreshScreenShot()
                    take_all_exist = task_window.FindButton('ACTION')
                    if (take_all_exist):
                        print('Кнопка "Забрать всё" найдена')
                        task_window.PressButton('ACTION')
                    else:
                        RandomMouseMove(treshold = 0.3)
                        task_window.action_button_template = t['take']
                        take_button_exist = True
                        while (take_button_exist):
                            task_window.FreshScreenShot()
                            take_button_exist = task_window.FindButton('ACTION')
                            if (take_button_exist):
                                task_window.PressButton('ACTION')

                        RandomMouseMove(treshold = 1)
                        task_window.action_button_template = t['start']
                        task_window.element_template = t['collapsed_elem']
                        task_window.FreshScreenShot()
                        start_button_exist = task_window.FindButton('ACTION', move_to = True)
                        collapsed_elem_exist = task_window.CheckElement()
                        # пока присутствует характерный значек свернутого задания
                        # или кнопка "начать", делаем рандомное количество кликов
                        i = 0
                        while (collapsed_elem_exist or start_button_exist):
                            print('collapsed_elem_exist=',collapsed_elem_exist)
                            i += 1
                            print('i =', i)
                            MultiClick(random.randint(4, 12))

                            task_window.FreshScreenShot()
                            task_window.element_template = t['collapsed_elem']
                            collapsed_elem_exist = task_window.CheckElement()
                            task_window.element_template = t['expanded_elem']
                            expanded_elem_exist = task_window.CheckElement()
                            if (i == 10):
                                task_window.FreshScreenShot()
                                start_button_exist = task_window.FindButton('ACTION', move_to = True)
                            # если значка свернутого задания нет, или есть
                            # значек развернутого задания, проверяем наличие кнопок "начать" и "забрать"
                            if (collapsed_elem_exist == False or expanded_elem_exist == True):
                                RandomMouseMove(treshold = 1)
                                task_window.FreshScreenShot()
                                start_button_exist = task_window.FindButton('ACTION', move_to = True)
                                # если нет кнопки "начать", проверяем наличие "забрать"
                                if (start_button_exist == False):
                                    RandomMouseMove(treshold = 1)
                                    task_window.action_button_template = t['take']
                                    start_button_exist = task_window.FindButton('ACTION', move_to = True)
                    if (current_number < number):
                        task_window.action_button_template = t['apply']
                        task_window.FreshScreenShot()
                        apply_button_exist = task_window.FindButton('ACTION')
                        if (apply_button_exist):
                            task_window.PressButton('ACTION')
                        else:
                            number = current_number

            task_window.FindButton('CLOSE', press = True)


    def SendResources(self, ReceiverName = r'coolrock\4'):
        print('SendResources start')
        zoom_template = './templates/zoom_panel.png'
        zoom_icon = Icon(self.screenshot_area, zoom_template, icon_treshold = 0.99)
        zoom_icon_exist = zoom_icon.Find()
        if (zoom_icon_exist == False):
            ScrollMouse(number = 12, direction = 'DOWN')
        else:
            print('Масштаб корректный')

        icon_template = './templates/town_market_s.png'
        icon_treshold = 0.7
        market_icon = Icon(self.screenshot_area, icon_template, icon_treshold)
        icon_exist = market_icon.Find()
        if (icon_exist):
            print('Market found')
            market_icon.Click()
            b_template = './templates/tab_market_help.png'
            market_window = Window(self.screenshot_area, b_template = b_template)
            help_tab_exist = market_window.FindButton('ACTION', press = True)
            if (help_tab_exist):
                print('Вкладка найдена и нажата')
                market_window.action_button_template = './templates/field_market.png'
                market_window.FreshScreenShot()
                market_window.FindButton('ACTION', press = True)
                #ReceiverName = r'coolrock\4'
                EnterText(text = ReceiverName)
                market_window.action_button_template = './templates/b_market_send_res.png'
                market_window.FreshScreenShot()
                market_window.FindButton('ACTION', press = True)
            else:
                market_window.FindButton('CLOSE', press = True)



    def TakeTechWorks(self):
        print('TakeTechWorks start')
        icon_template = './templates/icon_tech_works.png'
        icon_treshold = 0.7
        tw_icon = Icon(self.screenshot_area, icon_template, icon_treshold)
        icon_exist = tw_icon.Find()
        if (icon_exist):
            print('Icon exist')
            tw_icon.Click()
            b_template = './templates/b_take_tech_works.png'
            b_treshold = 0.5
            tw_window = Window(self.screenshot_area, b_template = b_template, b_treshold = b_treshold)
            button_exist = tw_window.FindButton('ACTION')
            if (button_exist):
                print('Кнопка найдена!')
                tw_window.PressButton('ACTION')
                print('button_exist =', button_exist)
            else:
                print('Кнопка не найдена!')

    def MultiBanUser(self, num = 1, close = True):
        print('MultiBanUser start at %s' %time.asctime())
        buttons = ('./templates/temp/b_ban_1.png',
                   './templates/temp/b_ban_2.png',
                   './templates/temp/b_unban_1.png',
                   './templates/temp/b_yes.png',
                   './templates/temp/b_close.png')
        usr_window = Window(self.screenshot_area,
                            b_treshold = 0.9)
        i = 0
        pressed = 0
        while (i != num):
            i += 1
            print('i =', i)
            for templ in buttons:
                print(templ)
                usr_window.action_button_template = templ
                button_exist = False

                # для кнопок из штатной последовательности делаем несколько попыток поиска
                # для нестандартной кнопки - только одна попытка
                j = 2 if (templ == buttons[4]) else 0
                while ((not button_exist) and (j < 3)):
                    j += 1

                    if ((templ == buttons[0]) or (templ == buttons[2])):
                        SleepRand(4, 6)

                    usr_window.FreshScreenShot()
                    button_exist = usr_window.FindButton('ACTION', press = True, delay = False, rand_mov_chance = 0.001)
                    if ((not button_exist) and (templ != buttons[4])):
                        print('Do not found button. Sleep and try again.')
                        SleepRand(acp_long[0], acp_long[1])
                    if (templ == buttons[0] and button_exist):
                        pressed += 1
                        print ('pressed =', pressed)

        print('*****************')
        print('TOTAL CLICKS: %s' %pressed)
        print('End time: %s' %time.asctime())
        print('*****************')
        if (close):
            x = 0
            while (x != 2):
                x += 1
                usr_window.FreshScreenShot()
                close_button_exist = usr_window.FindButton('CLOSE', press = True, rand_mov_chance = 0.1)
                RandomMouseMove(treshold = 1)
                print(close_button_exist)

    def RenameAcc(self, start = 1, end = 2):
        print('RenameAcc start')
        buttons = ('./templates/temp/b_edit.png',
                   './templates/temp/field_1.png',
                   './templates/temp/b_apply.png')
        profile_window = Window(self.screenshot_area,
                                b_treshold = 0.9)
        i = start-1
        while (i != end):
            i += 1
            print('i =', i)
            for templ in buttons:
                profile_window.action_button_template = templ
                button_exist = False
                j = 0
                while ((not button_exist) and (j < 3)):
                    j += 1
                    profile_window.FreshScreenShot()
                    #добавить условие и тип FIELD для правильного позиционирования курсора
                    button_exist = profile_window.FindButton('ACTION', press = True, delay = False, rand_mov_chance = 0.001)

                    if (not button_exist):
                        print('Do not found element. Sleep and try again.')
                        SleepRand(acp_long[0], acp_long[1])
                if (i > 1):
                    j = 0
                    while (j < len(str(i-1))):
                        j += 1
                        pg.hotkey('backspace')
                        SleepRand(0.1, 0.5)

                if (templ == buttons[1]):
                    SleepRand(acp_short[0], acp_short[1])
                    EnterText(text = str(i))





Init()
coolrock = Vik_akk()

SleepRand(acp_long[0], acp_long[1])
coolrock.MultiBanUser(90, close = False)
#coolrock.RenameAcc(start = 21, end = 46)

#coolrock.PromoClose()
#coolrock.TakeDailyLoyalBonus()
#coolrock.TakeChestOfLoki()
#coolrock.PressHelp()
#coolrock.TakeEveryDayBank()
#coolrock.TakeTasks(take_tab = 3, take_number = 1)
#coolrock.TakeTechWorks()
#coolrock.SendResources(r'Цита')
#coolrock.SendResources(r'CoolRock-4')



#recognize_image()
#file_name = CutThePict({'top': 0, 'left': 0, 'width': scr_resolution['x'], 'height': scr_resolution['y']}, png=True)
#file_name = CutThePict({'top': 247, 'left': 830, 'width': 500, 'height': 42}, png=True)
#scr_s = CutThePict({'top': 0, 'left': 0, 'width': scr_resolution['x'], 'height': scr_resolution['y']})
#print(scr_s)
