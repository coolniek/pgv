import numpy as np
import cv2
from mss.linux import MSS as mss
from mss.tools import to_png
from PIL import Image, ImageDraw, ImageChops
import time
import datetime as dttm
from datetime import datetime, date
import pyautogui as pg
import pytesseract
import os
import subprocess
import random
import math
import inspect

def Init():
    global acp_short, acp_long, lwp, debug, wc_time, load_time_treshold, scr_resolution
    global accounts, acc_icons, ren_click_delay, scr_center, chrome_path, chrome_key1, chrome_key2
    slow_machine = False
    #slow_machine = True

    chrome_path = '/usr/bin/google-chrome-stable'
    chrome_key1 = '%U'
    chrome_key2 = '--user-data-dir='
    # %U --user-data-dir="chrome/CoolRock2"
    if slow_machine:
        load_time_treshold = 150
        ren_click_delay = True
        acp_short = (0.6, 0.9)  # after click pause
        acp_long = (5, 6)       # after click pause
        lwp = 20                # load wait pause
        scr_resolution = {'x': 1280, 'y': 1024}
    else:
        load_time_treshold = 80
        ren_click_delay = False
        acp_short = (0.45, 0.7)
        #acp_long = (2, 4)
        acp_long = (1, 2)
        lwp = 10
        scr_resolution = {'x': 1920, 'y': 1080}
    debug = False
    #debug = True
    wc_time = 5000
    random.seed(version=2)
    scr_center = {'x': scr_resolution['x']/2, 'y': scr_resolution['y']/2+20}

    accounts = {'COOLROCK2': {'icon': './templates/desk_coolrock2.png',
                              'data_dir': '/home/istko/chrome/CoolRock2',
                              'resource': 'lumber',
                              'silver_prod': True,
                              'receiver': r'coolrock-4',
                              'default_mine': 'farm',
                              'default_marches': 13},
                'COOLROCK4': {'icon': './templates/desc_coolrock4.png',
                              'data_dir': '/home/istko/chrome/CoolRock4',
                              'resource': 'stone',
                              'silver_prod': False,
                              #'receiver': r'oxyir',
                              'receiver': r'COOLROCK\2',
                              'default_mine': 'iron',
                              'default_marches': 9},
                              #'receiver': r'coolrock\2'},
                'OXYIRON': {'icon': './templates/desk_oxyiron.png',
                            'data_dir': '/home/istko/chrome/OxyIron',
                            'resource': 'iron',
                            'silver_prod': True,
                            'receiver': r'coolrock-4',
                            'default_mine': 'stone',
                            'default_marches': 8},
                'FARAMEAR': {'icon': './templates/desk_faramear.png',
                             'data_dir': '/home/istko/chrome/Faramear',
                             'resource': 'food',
                             'silver_prod': True,
                             'receiver': r'coolrock-4',
                             'default_mine': 'lumber',
                             'default_marches': 9},
                'DUBROVSK': {'icon': './templates/desk_dubrovsk.png',
                             'data_dir': '/home/istko/chrome/Dubrovsk',
                             'resource': 'lumber',
                             'silver_prod': False,
                             'receiver': r'coolrock-4',
                             'default_mine': 'farm',
                             'default_marches': 5},
                'TRUVOR': {'icon': './templates/desk_truvor.png',
                           'data_dir': '/home/istko/chrome/Truvor',
                           'resource': 'stone',
                           'silver_prod': False,
                           'receiver': r'coolrock\2',
                           'default_mine': 'iron',
                           'default_marches': 4},
                'KRAKOZ': {'icon': './templates/desk_krakozyabrik.png',
                           'data_dir': '/home/istko/chrome/Krakozyabrik',
                           'resource': 'iron',
                           'silver_prod': False,
                           'receiver': r'oooo',
                           'default_mine': 'iron',
                           'default_marches': 3},
                'TEMPER': {'icon': './templates/desk_temperoker.png',
                           'data_dir': '/home/istko/chrome/Temperoker',
                           'resource': 'iron',
                           'silver_prod': False,
                           'receiver': r'coolrock-4',
                           'default_mine': 'iron',
                           'default_marches': 3}
                           }


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
#        if (debug):
#            print (x, y)
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


def MoveCurAndClick(x = 0, y = 0,
                    only_move = False,
                    x_slip = 5, y_slip = 5):
    print('x='+str(x)+', y='+str(y)) if (debug) else 0
    mv = {'x': x, 'y': y, 'tm': 0.5}
    min_x, max_x = x-x_slip, x+x_slip
    min_y, max_y = y-y_slip, y+y_slip
    mv['x'] = random.randint(min_x, max_x)
    mv['y'] = random.randint(min_y, max_y)
    #mv['tm'] = random.uniform(0.67, 0.9)
    mv['tm'] = random.uniform(0.33, 0.5)
    print(mv)

    MoveCursorRandPath(mv['x'], mv['y'], mv['tm'])
    SleepRand(acp_short[0], acp_short[1])
    if (only_move == False):
        ClickRandDelay()
        SleepRand(acp_short[0], acp_short[1])


def SleepRand(sleep_min=3, sleep_max=10):
    time.sleep(random.uniform(sleep_min, sleep_max))


def MultiClick(count=1):
    clicked = 0
    while (clicked < count):
        after_click_time = random.uniform(0.45, 0.9)
        ClickRandDelay()
        time.sleep(after_click_time)
        # немного смещаем курсор, если время задержки между кликами превысило определенный порог
        if (after_click_time > 0.85):
            x, y = pg.position()
            print('MultiClick. pg.position()=',pg.position())
            MoveCurAndClick(x, y, only_move=True, x_slip = 2, y_slip = 2)
        clicked += 1

# Функция для имитации человеческого присутсвия
# Двигает мышь в произвольное место в рамках заданного разрешения экрана
# Движение может быть выполнено от 0 до 2 раз.
# Количество определяется случайным образом в зависимости от указанной вероятности.
# По умолчанию вероятность 40%. Вероятность второй итерации уменьшается в 2 раза.
def RandomMouseMove(treshold = 0.4, times = 1):
    i = 0
    while (i != times):
        i += 1
        if (random.random() <= (treshold/i)):
            rand_x = random.randint(149, scr_resolution['x']-130)
            rand_y = random.randint(265, scr_resolution['y']-170)
            MoveCurAndClick(rand_x, rand_y, only_move = True)


def ScrollMouse(number = 10, direction = 'DOWN'):
    scrolled = 0
    interim = 0
    s_step = 1 if (direction == 'UP') else -1
    min_scroll, max_scroll = number - 2, number + 3
    target = random.randint(min_scroll, max_scroll)

    interval = random.randint(8, 11)
    while (scrolled < target):
        interim = scrolled + interval
        while ((scrolled < interim) and (scrolled < target)):
            pg.scroll(s_step)
            scrolled += 1
            print('Scrolling: {}'.format(scrolled), end='\r')
            if scrolled == target:
                print ("\n\r", end="")
            SleepRand(0.1, 0.2)
        SleepRand(0.3, 0.6)


def DragAndDrop(start_point, end_point):
    #start_point = {'x': 2, 'y': 4}
    #end_point = {'x': 12, 'y': 4}
    MoveCurAndClick(start_point['x'], start_point['y'], only_move=True)
    pg.mouseDown()
    SleepRand(acp_short[0], acp_short[1])
    MoveCurAndClick(end_point['x'], end_point['y'], only_move=True, x_slip = 30, y_slip = 10)
    SleepRand(acp_short[0], acp_short[1])
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
        print('"%s" is invalid parameter' %verifiable)
        print('Valid parameters are:')
        for l in valid_list:
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


def EnterText(text='coolrock\4', delay = True):
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
    if delay:
        SleepRand(acp_short[0], acp_short[1])


def CutThePict(area, png=False):
    area = area
    sct = mss()
    img = sct.grab(area)
    if (png == False):
        img_np = np.array(img)

        if debug:
            img_name = str(time.time())[:10]+str(time.time())[-2:]
            output = "scr/"+img_name+".png"
            to_png(img.rgb, img.size, output=output)

        return img_np
    else:
        img_name = str(time.time())[:10]+str(time.time())[-2:]
        output = "scr/"+img_name+".png"
        to_png(img.rgb, img.size, output=output)
        return output


def FindObject(full_img, templ_img, treshold):
    print('templ_img %s' % templ_img) if debug else 0
    position = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
    found = False   # флаг наличия изображения
    gray_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)   # преобразуем её в серую
    template = cv2.imread(templ_img, cv2.IMREAD_GRAYSCALE)  # искомый объект преобразуем в серый

    #cv2.imshow("img", gray_img)
    #cv2.waitKey(3000)
    #cv2.destroyAllWindows()

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


def ImgToBW(gray_img, scale = 5):
#cv.INTER_NEAREST — интерполяция методом ближайшего соседа (nearest-neighbor interpolation),
#cv.INTER_LINEAR — билинейная интерполяция (bilinear interpolation (используется по умолчанию),
#cv.INTER_CUBIC — бикубическая интерполяция (bicubic interpolation) в окрестности 4x4 пикселей,
#cv.INTER_AREA — передискретизации с использованием отношения площади пикселя,
#cv.INTER_LANCZOS4 — интерполяция Ланцоша (Lanczos interpolation) в окрестности 8x8 пикселей.
    gray_img = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    high = 255
    while(1):
        low = high - 1
        col_to_be_changed_low = np.array([low])
        col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(gray_img, col_to_be_changed_low, col_to_be_changed_high)
        if (high >= 160):
            gray_img[curr_mask > 0] = (255)
        else:
            gray_img[curr_mask > 0] = (0)
        high -= 1
        if(low == 0):
            break
    gray_img = cv2.bitwise_not(gray_img) # invert colors
    return gray_img


def grayscale_17_levels(gray_img):
    gray_img = cv2.resize(gray_img, None, fx=5, fy=5, interpolation = cv2.INTER_CUBIC)
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

def recognize_image(rec_area, lang = 'rus', scale = 5):
    img = CutThePict(rec_area)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (debug):
        cv2.imshow('image', img)
        cv2.waitKey(wc_time)
        cv2.destroyAllWindows()
    img = ImgToBW(img, scale = scale)
    if (debug):
        cv2.imshow('image', img)
        cv2.waitKey(wc_time)
        cv2.destroyAllWindows()
    #text='test'
    text = pytesseract.image_to_string(img, lang=lang)
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


def SwitchWindow():
    pg.keyDown('alt')
    time.sleep(1)
    pg.press('tab')
    time.sleep(1)
    pg.keyUp('alt')
    time.sleep(3)


def NowToStr(format = '%d.%m.%Y %H:%M:%S'):
    dt = datetime.now()
    str_date = dt.strftime(format)
    return str_date


def WriteLog(text, vars = {}, file_name = 'play.log'):
    params = ''
    if len(vars) > 0:
        for var_name, value in vars.items():
            params += '%s = %s, ' % (var_name, value)
        params += '\n'

    log = open(file_name, 'a')
    text += '\n'
    text = '%s %s%s' % (NowToStr(), text, params)
    log.write(text)
    log.close()


def SuspendSystem(sleep_time = 120, duration = 0):
    #add to /etc/sudoers:
    #%sudo	ALL=(ALL:ALL) NOPASSWD: /usr/sbin/pm-suspend
    #%sudo	ALL=(ALL:ALL) NOPASSWD: /usr/sbin/rtcwake
    FuncIntro(func_name = inspect.stack()[0][3], params = {'sleep_time':sleep_time}, only_print = False)

    print('Go to sleep in %s seconds...' %sleep_time)
    values = range(sleep_time, 0, -1)
    for i in values:
        print(' => %02d <=' %i, end='\r')
        time.sleep(1)
    #print ("\n\r", end="")
    if duration == 0:
        command = 'sudo pm-suspend'
    elif(duration > 0):
        command = 'sudo rtcwake -m mem -s '+str(duration)
    else:
        print('Bad duration parameter')
        return False

    os.system(command)


def FuncIntro(func_name, params = {}, only_print = True):
    log_text = '%s. ' % func_name
    print('%s %s' % (NowToStr(), log_text))
    if not only_print:
        WriteLog(log_text, vars = params)


def TextToResource(rec_text, production = True):
    FuncIntro(func_name = inspect.stack()[0][3])
    unit_scale = {'K': 1000,        # Latin symbols
                  'M': 1000000,
                  'B': 1000000000,
                  'К': 1000,        # Cyrilic symbols
                  'М': 1000000,
                  'В': 1000000000}
    err_text = ''
    amount = 0
    print('text: %s' %rec_text)
    text = rec_text.upper()
    text = text.replace(' ', '')
    text = text.replace(',', '.')
    if text == '':
        text = '0K'

    try:
        amount = float(text[:-1])
    except ValueError:
        err_text = 'Bad recognize: not digit symbols in amount.\n'
    except Exception:
        err_text = 'Undefined exception in TextToResource\n'
    else:
        unit = text[-1:]
        unit = 'B' if unit == '8' else unit
        point_position = text.find('.')
        #if production and (point_position == -1) and (unit in ('B', 'M', 'В', 'М')):
        #    amount = amount/10
        print('amount:', amount)
        amount *= unit_scale[unit]
        print('unit: %s' %unit)
        print('amount:', amount)
        #return int(amount)
    finally:
        if amount == 0:
            log_text = 'Amount: ' + str(amount) + ' Recognized text: ' + rec_text
            WriteLog(log_text, file_name = 'error.log')
        if err_text != '':
            err_text += ' Recognized text: ' + rec_text
            print(err_text)
            WriteLog(err_text, file_name = 'error.log')
        return int(amount)


class Window(object):

    def __init__(self,
                 shot_area,
                 h_substr = 'no_header',
                 h_template = 'no_header',
                 h_treshold = 0.5,
                 b_template = 'no_button',
                 b_treshold = 0.9,
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
        self.element_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
        self.slip = {'x': 3,'y': 3}

    def FreshScreenShot(self):
        print('FreshScreenShot') if debug else 0
        self.screen = CutThePict(self.shot_area)
        SleepRand(0.1, 0.3)

#    @staticmethod
    def CheckHeader(self, attempts = 3, lang = 'rus',
                    h_substr = 'no_header', h_template = 'no_header',
                    h_treshold = 0.5):
        print('CheckHeader start')
        if (self.header_substr == 'no_header'):
            return False
        header_exist = False
        count = 0
        while (not header_exist) and (count < attempts):
            count += 1
            self.FreshScreenShot()
            header_exist, header_border_pos = FindObject(self.screen, self.header_template, self.header_treshold)
        if (header_exist):
            n_top = header_border_pos['top_left'][1]
            n_right = header_border_pos['bottom_right'][0]
            n_height = (header_border_pos['bottom_right'][1])-(header_border_pos['top_left'][1])
            n_width = 500

            header_area = {'top': n_top, 'left': n_right, 'width': n_width, 'height': n_height}
            pos = -1
            count = 0
            while (pos == -1) and (count < attempts):
                count += 1
                rec_text = recognize_image(header_area, lang = lang)
                pos = rec_text.upper().find(self.header_substr.upper())
                print(self.header_substr)
                print('pos = '+str(pos))
                if (pos == -1):
                    print('Do not found header text. Sleep and try again.')
                    SleepRand(acp_short[0], acp_short[1])
            if (pos > -1):
                print('Слово "'+self.header_substr+'" найдено')
                return True
            else:
                return False
                print('Слово "'+self.header_substr+'" не найдено')
        else:
            print('Header not exist')
            return False

    def CheckElement(self, fresh_screen = False):
        self.FreshScreenShot() if (fresh_screen) else 0
        element_exist, self.element_pos = FindObject(self.screen, self.element_template, self.element_treshold)
        return element_exist

    def GetElementPos(self):
        return self.element_pos

    def FindUnit(self, type,
                 move_to = False,
                 press = False,
                 delay = True,
                 rand_mov_chance = 0.3,
                 field = False,
                 attempts = 3):
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

        button_exist = False
        count = 0
        while (not button_exist) and (count < attempts):
            count += 1
            self.FreshScreenShot()
            print('button_temlate = {}'.format(button_temlate))
            button_exist, button_pos = FindObject(self.screen, button_temlate, button_treshold)
            if ((not button_exist) and (count != attempts)):
                print('Do not found button. Sleep and try again.')
                RandomMouseMove(treshold = 1)
                SleepRand(acp_long[0], acp_long[1])

        #templ_width = button_pos['bottom_right'][0] - button_pos['top_left'][0]
        #templ_height = button_pos['bottom_right'][1] - button_pos['top_left'][1]
        #border_x = round((templ_width/100)*20)
        #border_y = round((templ_height/100)*20)
        #self.slip['x'] = round(templ_width/2)-border_x
        #self.slip['y'] = round(templ_height/2)-border_y if not field else 3
        self.slip = DefineSlip(button_pos, border = 20, field = field)


        if (type.upper() == 'ACTION'):
            self.action_button_pos = button_pos
        elif (type.upper() == 'CLOSE'):
            self.close_button_pos = button_pos

        if (button_exist and move_to):
            MoveCurAndClick(button_pos['center_point'][0], button_pos['center_point'][1],
                            only_move = True, x_slip = self.slip['x'], y_slip = self.slip['y'])

        if (button_exist and press):
            MoveCurAndClick(button_pos['center_point'][0], button_pos['center_point'][1],
                            x_slip = self.slip['x'], y_slip = self.slip['y'])
            if field:
                MoveCurAndClick(button_pos['center_point'][0], button_pos['center_point'][1],
                                x_slip = self.slip['x'], y_slip = self.slip['y'])
            RandomMouseMove(rand_mov_chance)
            if (delay == True):
                SleepRand(acp_long[0], acp_long[1])

        return button_exist

    def PressButton(self, type, delay = True, rand_mov_chance = 0.3):
        if (type.upper() == 'ACTION'):
            button_pos = self.action_button_pos
        elif (type.upper() == 'CLOSE'):
            button_pos = self.close_button_pos

        MoveCurAndClick(button_pos['center_point'][0], button_pos['center_point'][1],
                        x_slip = self.slip['x'], y_slip = self.slip['y'])
        RandomMouseMove(rand_mov_chance)
        if (delay == True):
            SleepRand(acp_long[0], acp_long[1])


class Icon(object):
    def __init__(self, shot_area, icon_template = 'no_template', icon_treshold = 0.7):
        """Constructor"""
        self.shot_area = shot_area
        self.icon_template = icon_template
        self.icon_treshold = icon_treshold
        self.icon_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
        self.slip = {'x': 3,'y': 3}

    def Find(self, click = False, attempts = 1, delay = True, rand_mov_chance = 0.3):
        count = 0
        icon_exist = False
        while (not icon_exist) and (count < attempts):
            count += 1
            self.screen = CutThePict(self.shot_area)
            icon_exist, self.icon_pos = FindObject(self.screen, self.icon_template, self.icon_treshold)
            if (not icon_exist) and (count < attempts):
                print('Do not found icon. Sleep and try again. count = %s' %count)
                RandomMouseMove(treshold = 1)
                SleepRand(acp_short[0], acp_short[1])

        if icon_exist and click:
            self.Click(delay = delay, rand_mov_chance = rand_mov_chance)
        return icon_exist

    def CheckNumber(self, number_color_range, icon_attempts = 1):
        #number_color_range = {'low_col': (18,11,85), 'high_col': (38,25,151)} (BGR)
        count = 0
        icon_exist = False
        while (not icon_exist) and (count < icon_attempts):
            count += 1
            self.screen = CutThePict(self.shot_area)
            icon_exist, self.icon_pos = FindObject(self.screen, self.icon_template, self.icon_treshold)
            if (not icon_exist) and (count < icon_attempts):
                print('Do not found icon. Sleep and try again.')
                SleepRand(acp_short[0], acp_short[1])
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


class WindowArea(object):
    def __init__(self, shot_area,
                 left_border_templ = 'no_template',
                 right_border_templ = 'no_template',
                 top_border_templ = 'no_template',
                 bottom_border_templ = 'no_template',
                 action_elem_templ = 'no_template',
                 left_border_treshold = 0.9,
                 right_border_treshold = 0.9,
                 top_border_treshold = 0.9,
                 bottom_border_treshold = 0.9,
                 action_elem_treshold = 0.9
                 ):
        self.shot_area = shot_area
        self.left_border_templ = left_border_templ
        self.left_border_treshold = left_border_treshold
        self.right_border_templ = right_border_templ
        self.right_border_treshold = right_border_treshold
        self.top_border_templ = top_border_templ
        self.top_border_treshold = top_border_treshold
        self.bottom_border_templ = bottom_border_templ
        self.bottom_border_treshold = bottom_border_treshold
        self.action_elem_templ = action_elem_templ
        self.action_elem_treshold = action_elem_treshold
        self.left_border_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
        self.right_border_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
        self.top_border_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
        self.bottom_border_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
        self.action_elem_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
        self.action_elem_coord = {'x': 0, 'y': 0}
        self.search_area = {'top': 0, 'left': 0, 'width': 0, 'height': 0}
        self.area_defined = False
        self.action_elem_exist = False
        self.screen = CutThePict(self.shot_area)
        self.left, self.top, self.width, self.height = 0, 0, 0, 0

    def DefineArea(self, direction = 'left_to_right', attempts = 3, scroll = 10):
        print('DefineArea start')
        left_border_exist = False
        right_border_exist = False
        top_border_exist = False
        bottom_border_exist = False
        print('self.left_border_templ %s' % self.left_border_templ)
        print('self.right_border_templ %s' % self.right_border_templ)

        if (direction == 'left_to_right'):
            count = 0
            while (not right_border_exist) and (count < attempts):
                count += 1
                self.screen = CutThePict(self.shot_area)
                right_border_exist, self.right_border_pos = FindObject(self.screen,
                                                                    self.right_border_templ,
                                                                    self.right_border_treshold)
                if (not right_border_exist) and (count < attempts):
                    print('Do not found border. Sleep and try again. count = %s' %count)
                    RandomMouseMove(treshold = 1)
                    SleepRand(acp_short[0], acp_short[1])
            if not right_border_exist:
                print('Right border not found')
                return False


            count = 0
            page = 0
            while (not left_border_exist) and (page < 10):
                count += 1
                self.screen = CutThePict(self.shot_area)
                left_border_exist, self.left_border_pos = FindObject(self.screen,
                                                                    self.left_border_templ,
                                                                    self.left_border_treshold)
                if (not left_border_exist) and (count == attempts):
                    count = 0
                    MoveCurAndClick(self.right_border_pos['center_point'][0],
                                    self.right_border_pos['center_point'][1],
                                    only_move = True)
                    ScrollMouse(number = scroll, direction = 'DOWN')
                    page += 1


            self.left = self.left_border_pos['bottom_right'][0]
            self.top = self.left_border_pos['top_left'][1]
            self.width = self.right_border_pos['top_left'][0] - self.left
            self.height = self.left_border_pos['bottom_right'][1] - self.top
        elif (direction == 'right_to_left'):
            count = 0
            while (not left_border_exist) and (count < attempts):
                count += 1
                self.screen = CutThePict(self.shot_area)
                left_border_exist, self.left_border_pos = FindObject(self.screen,
                                                                     self.left_border_templ,
                                                                     self.left_border_treshold)
            if not left_border_exist:
                print('Right border not found')
                return False

            count = 0
            page = 0
            while (not right_border_exist) and (page < 10):
                count += 1
                self.screen = CutThePict(self.shot_area)
                right_border_exist, self.right_border_pos = FindObject(self.screen,
                                                                    self.right_border_templ,
                                                                    self.right_border_treshold)
                if (not right_border_exist) and (count == attempts):
                    count = 0
                    MoveCurAndClick(self.left_border_pos['center_point'][0],
                                    self.left_border_pos['center_point'][1],
                                    only_move = True)
                    ScrollMouse(number = scroll, direction = 'DOWN')
                    page += 1


            self.left = self.left_border_pos['bottom_right'][0]
            self.top = self.right_border_pos['top_left'][1]
            self.width = self.right_border_pos['top_left'][0] - self.left
            self.height = self.right_border_pos['bottom_right'][1] - self.top
        elif (direction == 'top_to_bottom'):
            print('direction == \'top_to_bottom\'')
            count = 0
            while (not bottom_border_exist) and (count < attempts):
                count += 1
                self.screen = CutThePict(self.shot_area)
                bottom_border_exist, self.bottom_border_pos = FindObject(self.screen,
                                                                    self.bottom_border_templ,
                                                                    self.bottom_border_treshold)
            if not bottom_border_exist:
                print('Bottom border not found')
                return False

            count = 0
            while (not top_border_exist) and (count < attempts):
                count += 1
                self.screen = CutThePict(self.shot_area)
                top_border_exist, self.top_border_pos = FindObject(self.screen,
                                                                    self.top_border_templ,
                                                                    self.top_border_treshold)

            self.left = self.top_border_pos['top_left'][0]
            self.top = self.top_border_pos['bottom_right'][1]
            self.width = self.top_border_pos['bottom_right'][0] - self.left
            self.height = self.bottom_border_pos['top_left'][1] - self.top

        self.search_area = {'top': self.top,
                            'left': self.left,
                            'width': self.width,
                            'height': self.height}
        left_to_right_exist = (direction in ('left_to_right', 'right_to_left') and left_border_exist and right_border_exist)
        top_to_bottom_exist = (direction == 'top_to_bottom' and top_border_exist and bottom_border_exist)
        if left_to_right_exist or top_to_bottom_exist:
            self.area_defined = True
        return self.area_defined

    def FindElem(self,
                 action_elem_templ = 'no_template',
                 action_elem_treshold = 0,
                 attempts = 3):
        print('FindElem start')
        print('self.action_elem_templ: %s' % self.action_elem_templ)
        print('action_elem_templ: %s' % action_elem_templ)
        if not self.area_defined:
            return False
        if (action_elem_templ != 'no_template'):
            self.action_elem_templ = action_elem_templ
        if (action_elem_treshold != 0):
            self.action_elem_treshold = action_elem_treshold
        self.action_elem_exist = False
        count = 0
        while (not self.action_elem_exist) and (count < attempts):
            count += 1
            area_screen = CutThePict(self.search_area)
            self.action_elem_exist, self.action_elem_pos = FindObject(area_screen,
                                                                 self.action_elem_templ,
                                                                 self.action_elem_treshold)
            if (not self.action_elem_exist) and (count < attempts):
                print('Do not found element. Sleep and try again. count = %s' %count)
                RandomMouseMove(treshold = 1)
                SleepRand(acp_short[0], acp_short[1])

        self.action_elem_coord['x'] = self.left + self.action_elem_pos['center_point'][0]
        self.action_elem_coord['y'] = self.top + self.action_elem_pos['center_point'][1]
        return self.action_elem_exist

    def ClickElem(self, delay = True, rand_mov_chance = 0.3, field = False):
        if not self.action_elem_exist:
            return False
        slip = DefineSlip(self.action_elem_pos, border = 20, field = field)
        MoveCurAndClick(self.action_elem_coord['x'], self.action_elem_coord['y'],
                        x_slip = slip['x'], y_slip = slip['y'])
        RandomMouseMove(rand_mov_chance)
        if (delay == True):
            SleepRand(acp_long[0], acp_long[1])
        return True

    def DragToRightBorder(self):
        if not self.action_elem_exist:
            return False
        drag_end_point = {'x': 0, 'y': 0}
        drag_end_point['x'] = self.right_border_pos['center_point'][0]
        drag_end_point['y'] = self.left_border_pos['center_point'][1]

        #MoveCurAndClick(action_elem_pos['x'], action_elem_pos['y'],
        #                only_move = True, x_slip = 3, y_slip = 3)
        DragAndDrop(start_point = self.action_elem_coord, end_point = drag_end_point)
        SleepRand(acp_short[0], acp_short[1])

    def RecognizeText(self, lang = 'eng', pict_scale = 5):
        if not self.area_defined:
            return ''
        print('RecognizeText start')
        rec_text = recognize_image(self.search_area, lang = lang, scale = pict_scale)
        return rec_text



class Vik_acc():
    login = ''
    def __init__(self, account_name = 'no_account'):
        self.screenshot_area = {'top': 0, 'left': 0, 'width': scr_resolution['x'], 'height': scr_resolution['y']}
        self.red_number_color_range = {'low_col': (18,11,85), 'high_col': (38,25,151)}
        self.account_name = account_name.upper()


    def LogOn(self):
        units = ('./templates/site/b_logon.png',
                 './templates/site/remember_checkbox.png',
                 './templates/site/b_logon_2.png',
                )
        for templ in units:
            window = Window(self.screenshot_area)
            window.action_button_template = templ
            unit_exist = window.FindUnit('ACTION', press = True, attempts = 2, delay = False, rand_mov_chance = 0)
            #if unit_exist:
            #    continue
        return unit_exist



    def StartGame(self, from_desktop = True):
        FuncIntro(func_name = inspect.stack()[0][3], params = {'account_name': self.account_name,'from_desktop': from_desktop}, only_print = False)

        game_started = False
        load_attempts = 7
        attempt = 0
        load_template = './templates/viki_load_logo.png'

        load_window = Window(self.screenshot_area, b_template = load_template)
        silver_icon = Icon(self.screenshot_area,
                           icon_template = './templates/icon_panel_silver.png',
                           icon_treshold = 0.99)
        window_existed = self.CloseAllWindows()
        silver_icon_exist = silver_icon.Find()
        if silver_icon_exist or window_existed:
            print('Game is running')
            game_started = True
            return True

        if not game_started and from_desktop:
            print('not game_started and from_desktop')
            valid_acc = ValidParam(accounts, self.account_name)
            if not valid_acc:
                print('Account name invalid')
                return False
            #acc_icons[self.account_name]
            desk_icon = Icon(self.screenshot_area,
                             icon_template = accounts[self.account_name]['icon'],
                             icon_treshold = 0.99)
            desk_icon_exist = desk_icon.Find(click = True)
            if desk_icon_exist:
                pg.hotkey('enter')
                time.sleep(3)
            else:
                print('StartGame - Popen')
                command = (chrome_path, chrome_key1, chrome_key2+accounts[self.account_name]['data_dir'])
                print(command)
                #p = subprocess.Popen(['/usr/bin/google-chrome-stable'])
                subprocess.Popen(command)

        SleepRand(7, 10)


        while (not game_started) and (attempt < load_attempts):
            print('Attempt: %s' %attempt)
            attempt += 1
            start_time = datetime.now()
            current_time = datetime.now()
            delta = current_time - start_time

            logon = self.LogOn()
            if logon:
                print('StartGame - logon')
                self.CloseGame()
                self.StartGame(from_desktop = True)

            logo_exist = load_window.FindUnit(type = 'ACTION', press = True, attempts = 10)
            while (logo_exist and (delta.seconds < load_time_treshold)):
                logo_exist = load_window.FindUnit(type = 'ACTION', attempts = 1)
                if logo_exist:
                    print('Loading. Waiting %s seconds...' %lwp)
                    time.sleep(lwp)
                else:
                    print('Logo not exist')
                current_time = datetime.now()
                delta = current_time - start_time
                print('%s seconds have passed since the beginning.' %delta.seconds)

            SleepRand(acp_long[0]+2, acp_long[1]+2)
            game_started = self.CloseAllWindows()
            print('game_started = %s' %game_started)
            ScrollMouse(number = 11, direction = 'DOWN') if game_started else 0

            if (logo_exist or (not logo_exist and not game_started)) and (attempt < load_attempts):
                if attempt <= 3:
                    print('Press F5')
                    pg.hotkey('f5')
                elif attempt > 3:
                    print('Press Ctrl+F5')
                    pg.keyDown('ctrl')
                    time.sleep(1)
                    pg.press('f5')
                    time.sleep(1)
                    pg.keyUp('ctrl')

        if not game_started:
            print('Game start error')
            self.CloseGame()
            return False
        else:
            print('Game started')
            return True


    def CloseGame(self):
        FuncIntro(func_name = inspect.stack()[0][3],
                  params = {'account_name': self.account_name},
                  only_print = False)
        buttons = ('./templates/b_browser_menu.png',
                   './templates/b_browser_exit.png')
        browser_window = Window(self.screenshot_area,
                                b_treshold = 0.9)

        for templ in buttons:
            browser_window.action_button_template = templ
            button_exist = browser_window.FindUnit('ACTION', press = True, rand_mov_chance = 0.001)

        SleepRand(7, 10)


    def CloseAllWindows(self):
        func_name = inspect.stack()[0][3]
        FuncIntro(func_name)
        window_existed = False
        window = Window(self.screenshot_area,
                              e_template = './templates/header_border_right.png',
                              e_treshold = 0.5)
        count = 0
        window_exist = window.CheckElement(fresh_screen = True)
        while window_exist and (count < 5):
            count += 1
            window_existed = True
            window_closed = window.FindUnit('CLOSE', press = True)
            if not window_closed:
                print('Press Ctrl+F5')
                pg.keyDown('ctrl')
                time.sleep(1)
                pg.press('f5')
                time.sleep(1)
                pg.keyUp('ctrl')
            RandomMouseMove(treshold = 1)
            window_exist = window.CheckElement(fresh_screen = True)
        print('{} window_exist = {}'.format(window_exist, func_name))
        print('count = %s' %count)
        if window_exist and (count == 5):
            print('Отсутствует кнопка "Закрыть"')
            return window_closed
        return window_existed


    def ToScale(self):
        RandomMouseMove(treshold = 1)
        zoom_icon = Icon(self.screenshot_area,
                         icon_template = './templates/zoom_panel.png',
                         icon_treshold = 0.99)
        zoom_icon_exist = zoom_icon.Find()


        if (zoom_icon_exist == False):
            ScrollMouse(number = 12, direction = 'DOWN')
        else:
            ScrollMouse(number = 3, direction = 'DOWN')
            print('The scale is correct')

    def TakeDailyLoyalBonus(self):
        FuncIntro(func_name = inspect.stack()[0][3])
        dl_icon = Icon(self.screenshot_area,
                       icon_template = './templates/icon_daily_loyal.png',
                       icon_treshold = 0.5)
        icon_exist = dl_icon.Find(attempts = 2)
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
                button_exist = dl_window.FindUnit('ACTION', press = True)
            else:
                print('Необходимое окно не открылось')
                self.StartGame(from_desktop = True)
                # добавить закрытие окна
        else:
            print('No icon')

# вынести повторяющийся код в виде атрибутов
# сделать универсальный метод для статичных иконок/окон, если это будет целесообразно (например дневная лояльность и помощь)
# разделить кнопки на "забрать" и "закрыть"
    def PressHelp(self, help_mode = False):
        FuncIntro(func_name = inspect.stack()[0][3])
        rand_mov_chance = 0.3
        count = 0
        hlp_icon = Icon(self.screenshot_area,
                        icon_template = './templates/icon_clan_help.png',
                        icon_treshold = 0.7)
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
                if help_mode:
                    rand_mov_chance = 1
                while (count < 1):
                    count += 1 if not help_mode else 0
                    button_exist = hlp_window.FindUnit('ACTION', press = True, rand_mov_chance = rand_mov_chance)
                    SleepRand(2, 3) if help_mode else 0
                close_button_exist = hlp_window.FindUnit('CLOSE', press = True)
            else:
                self.StartGame(from_desktop = False)
        else:
            print('No icon')


    def TakeEveryDayBank(self):
        FuncIntro(func_name = inspect.stack()[0][3])
        icon_number_cr = self.red_number_color_range
        bank_icon = Icon(self.screenshot_area,
                         icon_template = './templates/icon_bank_part.png',
                         icon_treshold = 0.9)
        icon_exist = bank_icon.Find()
        if not icon_exist:
            self.StartGame(from_desktop = False)
        icon_numbered = bank_icon.CheckNumber(icon_number_cr, icon_attempts = 3)
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
            button_exist = bank_window.FindUnit('ACTION')
            print('header_exist =', header_exist)
            print('button_exist =', button_exist)
            if (header_exist and button_exist):
                bank_window.PressButton('ACTION')
                print('Subscribe tab pressed')
                bank_window.action_button_template = './templates/b_take_3_bank1.png'
                bank_window.action_button_treshold = 0.8
                button_exist = bank_window.FindUnit('ACTION')
                if (button_exist):
                    bank_window.PressButton('ACTION')
                    bank_window.action_button_template = './templates/b_take_4_bank2.png'
                    button_exist = bank_window.FindUnit('ACTION', press = True)
                else:
                    print('Кнопка "забрать" не обнаружена')
                    self.StartGame(from_desktop = False)
                close_button_exist = bank_window.FindUnit('CLOSE', press = True)
            else:
                print('Заголовок не соответствует или вкладка не обнаружена')
                close_button_exist = bank_window.FindUnit('CLOSE', press = True)
                self.StartGame(from_desktop = False)
        else:
            print('Icon NOT numbered')


    def TakeChestOfLoki(self):
        FuncIntro(func_name = inspect.stack()[0][3])
        chol_icon = Icon(self.screenshot_area,
                         icon_template = './templates/icon_loki_chest.png',
                         icon_treshold = 0.7)
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
                                     b_treshold = 0.9)
                button_exist = chol_window.FindUnit('ACTION', press = True)
                print('button_exist = %s' % button_exist)
            else:
                print('No movement')
        else:
            print('Chest of Loki not found')
            self.StartGame(from_desktop = False)


    def TakeQuests(self):
        FuncIntro(func_name = inspect.stack()[0][3])
        town_icons = ('templates/quests/town_white_icon.png',
                      'templates/quests/town_green_icon.png',
                      'templates/quests/town_blue_icon.png',
                      'templates/quests/town_purple_icon.png',
                      )
        quests_icons = ('templates/quests/quests_white_icon.png',
                        'templates/quests/quests_green_icon.png',
                        'templates/quests/quests_blue_icon.png',
                        'templates/quests/quests_purple_icon.png',
                        )
        buttons = {'take':  'templates/quests/take_b.png',
                   'close': 'templates/quests/close_b.png'}

        quests_icon = Icon(self.screenshot_area, icon_treshold = 0.7)
        for templ in town_icons:
            quests_icon.icon_template = templ
            icon_exist = quests_icon.Find()
            if icon_exist:
                break
        if icon_exist:
            print('Quest icon found')
            quests_icon.Click()
        else:
            return False

        quests_window = Window(self.screenshot_area)
        for templ in quests_icons:
            quests_window.action_button_template = templ
            quests_window.action_button_treshold = 0.99
            button_exist = quests_window.FindUnit('ACTION', press = True, delay = False, attempts = 1)
            if button_exist:
                for key, templ in buttons.items():
                    quests_window.action_button_template = templ
                    quests_window.FindUnit('ACTION', press = True, delay = True, attempts = 1)

        quests_window.FindUnit('CLOSE', press = True, delay = False, attempts = 2)


# Проверяем наличие новых заданий (число красном кружочке) и жмем на иконку при наличии
# Так же жмем, если необходимо выполнить задания многократно (применяя обновление заданий)
# Проверяем наличие кнопки "Забрать все" и жмем ее при наличии
# Если этой кнопки нет, ищем кнопку начать, перемещаемся на нее и прокликиваем.
# Так для каждой вкладки
# Может быть запрошено выполнение нескольких циклов заданий для какой-то из вкладок
# take_tab - порядковый номер вкладки, для которой запрошено несколько заданий
# take_number - требуемое количество циклов выполнения заданий для запрошенной вкладки
    def TakeTasks(self, take_tab = 1, take_number = 1, start_game = False, close_game = False):
        FuncIntro(func_name = inspect.stack()[0][3])

        if start_game:
            self.StartGame(from_desktop = True)
        icon_number_cr = self.red_number_color_range
        task_icon = Icon(self.screenshot_area,
                         icon_template = './templates/icon_tasks.png',
                         icon_treshold = 0.7)
        icon_exist = task_icon.Find()
        if not icon_exist:
            self.StartGame(from_desktop = False)
        icon_numbered = task_icon.CheckNumber(icon_number_cr)
        if (icon_numbered or take_number > 1):
            print('Icon numbered')
            task_icon.Click(rand_mov_chance = 0)
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
            total_applied = 0
            for tab in tabs:
                current_tab += 1
                current_number = 0
                number = take_number if (current_tab == take_tab) else 1
                if (tab != tabs[0]):
                    task_window.action_button_template = tab
                    task_window.action_button_treshold = 0.7
                    tab_exist = task_window.FindUnit('ACTION', press = True, rand_mov_chance = 0)
                    if not tab_exist:
                        self.StartGame(from_desktop = False)
                        task_window.FindUnit('CLOSE', press = True)
                        return 1
                applied = 0
                while (current_number < number):
                    current_number += 1
                    task_window.action_button_template = t['take_all']
                    task_window.action_button_treshold = 0.99
                    take_all_exist = task_window.FindUnit('ACTION', attempts = 1)
                    if (take_all_exist):
                        print('Кнопка "Забрать всё" найдена')
                        task_window.PressButton('ACTION', rand_mov_chance = 0.0001, delay = False)
                    else:
                        task_window.action_button_template = t['take']
                        take_button_exist = True
                        while (take_button_exist):
                            take_button_exist = task_window.FindUnit('ACTION',
                                                                     press = True,
                                                                     attempts = 1,
                                                                     rand_mov_chance = 1,
                                                                     delay = False)

                        task_window.action_button_template = t['start']
                        task_window.element_template = t['collapsed_elem']
                        start_button_exist = task_window.FindUnit('ACTION', move_to = True, attempts = 1)
                        collapsed_elem_exist = task_window.CheckElement()
                        expanded_elem_exist = False
                        # пока присутствует характерный значок свернутого задания
                        # или кнопка "начать", делаем рандомное количество кликов
                        i = 0
                        while (collapsed_elem_exist or start_button_exist or expanded_elem_exist):
                            print('collapsed_elem_exist=',collapsed_elem_exist)
                            i += 1
                            print('i =', i)
                            base = 17
                            clicks_avg = base - i*2 if ((base - i*2) >= 5) else 5
                            click_range = (clicks_avg-2, clicks_avg+2)
                            MultiClick(random.randint(click_range[0], click_range[1]))

                            task_window.element_template = t['collapsed_elem']
                            collapsed_elem_exist = task_window.CheckElement(fresh_screen = True)
                            task_window.element_template = t['expanded_elem']
                            expanded_elem_exist = task_window.CheckElement()
                            task_window.action_button_template = t['start']
                            start_button_exist = task_window.FindUnit('ACTION', attempts = 1)
                            # если значка свернутого задания нет,
                            # или есть значок развернутого задания,
                            # проверяем наличие кнопок "забрать" и "начать"
                            #only_expanded_exist = collapsed_elem_exist == False and expanded_elem_exist == True
                            too_much_clicks = collapsed_elem_exist == True and i >= 4
                            if expanded_elem_exist or too_much_clicks:
                                task_window.action_button_template = t['take']
                                take_button_exist = task_window.FindUnit('ACTION', press = True, attempts = 2, rand_mov_chance = 0.0001)
                                task_window.action_button_template = t['start']
                                start_button_exist = task_window.FindUnit('ACTION', move_to = True, attempts = 2)
                                task_window.element_template = t['expanded_elem']
                                expanded_elem_exist = task_window.CheckElement(fresh_screen = True)

                    if (current_number < number):
                        task_window.action_button_template = t['apply']
                        apply_button_exist = task_window.FindUnit('ACTION', attempts = 2)
                        if (apply_button_exist):
                            task_window.PressButton('ACTION', rand_mov_chance = 0.0001)
                            applied += 1
                            print('Applied %s task refreshes' %applied)
                        else:
                            number = current_number
                            self.StartGame(from_desktop = False)
                print('APPLIED FOR TAB: %s' %applied)
                total_applied += applied
            print('*****************')
            print('TOTAL APPLIED: %s' %total_applied)
            print('End time: %s' %time.asctime())
            print('*****************')
            task_window.FindUnit('CLOSE', press = True)
            if close_game:
                self.CloseGame()


    def SendResources(self, receiver_name = r'', res_list = [], production = True, send_amount = ''):
        FuncIntro(func_name = inspect.stack()[0][3])
        area_exist = False
        slider_exist = False
        sent_success = False
        self.StartGame(from_desktop = False)
        self.ToScale()

        icon_templates = ('./templates/town_market_1080_21lvl.png',
                          './templates/town_market_1080_22lvl.png',
                          './templates/town_market_1024_23lvl.png',
                          './templates/town_market_1080_27lvl.png',
                          './templates/town_market_1024_27lvl.png')
        units = {'market_help_tab': './templates/tab_market_help.png',
                 'field':           './templates/field_market.png',
                 'send_res':        './templates/b_market_send_res.png',
                 'send_area_border':'./templates/border_market_send.png',
                 'res_panel_border':'./templates/border_panel_res.png',
                 'max_res_border_top':'./templates/border_max_res_top.png',
                 'max_res_border_bottom':'./templates/border_max_res_bottom.png',
                 'slider':          './templates/b_market_slider.png',
                 'send':            './templates/b_market_send.png',
                 'close':            './templates/market/b_market_close.png'}
        market_res_icons = {'food':    './templates/icon_market_food.png',
                            'lumber':  './templates/icon_market_lumber.png',
                            'iron':    './templates/icon_market_iron.png',
                            'stone':   './templates/icon_market_stone.png',
                            'silver':  './templates/icon_market_silver.png'}
        panel_res_icons = {'food':    './templates/icon_panel_food.png',
                           'lumber':  './templates/icon_panel_lumber.png',
                           'iron':    './templates/icon_panel_iron.png',
                           'stone':   './templates/icon_panel_stone.png',
                           'silver':  './templates/icon_panel_silver.png'}

        if (receiver_name == ''):
            receiver_name = accounts[self.account_name]['receiver']
        if (len(res_list) == 0):
            res_list.append(accounts[self.account_name]['resource'])
            if (accounts[self.account_name]['silver_prod'] == True):
                res_list.append('silver')
        print('accounts[self.account_name][resource]: %s' %accounts[self.account_name]['resource'])
        print('self.account_name: %s' %self.account_name)
        print('res_list: %s' %res_list)

        # Определяем количество ресурсов в городе
        res_amount = {}
        amount_text = ''
        for res in res_list:
            res_pan = WindowArea(self.screenshot_area,
                                 left_border_templ = panel_res_icons[res],
                                 right_border_templ = units['res_panel_border'])

            count = 0
            res_area_exist = False
            while (not res_area_exist) and (count < 2):
                count += 1
                res_area_exist = res_pan.DefineArea(direction = 'left_to_right')
                if not res_area_exist:
                    self.CloseAllWindows()
            if res_area_exist:
                amount_text = res_pan.RecognizeText(lang = 'eng')
                #print('%s, amount %s' (res, amount_text))
                print('Resource: %s' %res)
                print('amount_text: %s' %amount_text)
                amount = TextToResource(amount_text)
                res_amount[res] = {'amount': amount,
                                   'caravans': 0}
        print('res_amount:', res_amount)
        #return 0


        # Ищем рынок
        market_icon = Icon(self.screenshot_area, icon_treshold = 0.7)
        for templ in icon_templates:
            market_icon.icon_template = templ
            icon_exist = market_icon.Find()
            if (icon_exist):
                break
        if (icon_exist):
            print('Market found')
            market_icon.Click()
            b_template = units['market_help_tab']
            market_window = Window(self.screenshot_area, b_template = b_template)
            help_tab_exist = market_window.FindUnit('ACTION', press = True)
            if help_tab_exist:
                print('Tab found and pressed')
                market_window.action_button_template = units['field']
                market_window.action_button_treshold = 0.9
                market_window.FindUnit('ACTION', press = True, field = True)
                EnterText(text = receiver_name)
                market_window.action_button_template = units['send_res']
                button_exist = market_window.FindUnit('ACTION', press = True)
                if button_exist:
                    max_send = 0
                    # Определяем вместимость караванов и количество к отправке
                    max_area = WindowArea(self.screenshot_area,
                                     top_border_templ = units['max_res_border_top'],
                                     bottom_border_templ = units['max_res_border_bottom'])
                    max_area_exist = max_area.DefineArea(direction = 'top_to_bottom')
                    if not max_area_exist:
                        print('Not max_area_exist')
                        self.StartGame(from_desktop = False)
                        return False
                    count = 0
                    #CountDown(3)
                    pict_scale = 5
                    while (max_send == 0) and (count < 3):
                        count += 1
                        amount_text = max_area.RecognizeText(lang = 'eng', pict_scale = pict_scale)
                        max_send = TextToResource(amount_text)
                        if max_send == 0:
                            pict_scale += 1
                            RandomMouseMove(treshold = 1)
                    print('Max resource send:', max_send)
                    if max_send == 0:
                        self.StartGame(from_desktop = False)
                        return False

                    # Ищем нужные ползунки и отправляем ресурсы
                    sent_success = False
                    for resource in res_list:
                        res_amount[resource]['caravans'] = math.ceil(res_amount[resource]['amount']/max_send)
                        if (production and res_amount[resource]['caravans'] > 10):
                            res_amount[resource]['caravans'] = 10
                        print('res_amount:', res_amount)
                        number_of_caravans = 2 if (resource == 'silver') else 4
                        res = WindowArea(self.screenshot_area,
                                         left_border_templ = market_res_icons[resource],
                                         right_border_templ = units['send_area_border'],
                                         action_elem_templ = units['slider'])
                        area_exist = res.DefineArea(direction = 'left_to_right')
                        if area_exist:
                            slider_exist = res.FindElem()
                        if slider_exist:
                            count = 0
                            while (count < res_amount[resource]['caravans']):
                            #while (count < number_of_caravans):
                                count += 1
                                res.DragToRightBorder()
                                market_window.action_button_template = units['send']
                                button_exist = market_window.FindUnit('ACTION',
                                                                      press = True,
                                                                      delay = False,
                                                                      rand_mov_chance = 0)
                                market_window.action_button_template = units['close']
                                close_button_exist = market_window.FindUnit('ACTION',
                                                                      press = True,
                                                                      delay = False,
                                                                      attempts = 1,
                                                                      rand_mov_chance = 0)
                                if close_button_exist:
                                    count -= 1
                                if not button_exist:
                                    break
                                    self.StartGame(from_desktop = False)
                        else:
                            print('Slider not found')
                            self.StartGame(from_desktop = False)
                        if count == res_amount[resource]['caravans']:
                            sent_success = True
                        else:
                            sent_success = False
                else:
                    print('Button not found')
                    self.StartGame(from_desktop = False)

            else:
                market_window.FindUnit('CLOSE', press = True)
            self.CloseAllWindows()
            return sent_success
        else:
            print('Market not found!')
            self.StartGame(from_desktop = False)

        return False


    def SendToMine(self, need_type = 'farm', need_level = 'lvl6', number_of_troops = '1111'):
        FuncIntro(func_name = inspect.stack()[0][3])
        icons = ('./templates/tab_panel_1.png',
                 './templates/icon_panel_map.png',
                 './templates/map_navigator.png'
                 )
        types = {'select':       './templates/navigator/navi_menu_type_select.png',
                 'farm':         './templates/navigator/navi_menu_type_farm.png',
                 'invader_lair': './templates/navigator/navi_menu_type_invader_lair.png',
                 'lumber':       './templates/navigator/navi_menu_type_lumber.png',
                 'iron':         './templates/navigator/navi_menu_type_iron.png',
                 'stone':        './templates/navigator/navi_menu_type_stone.png',
                 'silver':       './templates/navigator/navi_menu_type_silver.png',
                 'invader':      './templates/navigator/navi_menu_type_invader.png',
                 'ghost':        './templates/navigator/navi_menu_type_ghost.png',
                 'ghost_shelter':'./templates/navigator/navi_menu_type_ghost_shelter.png'}

        levels = {'lvl6':         './templates/navigator/navi_menu_level_6.png',
                  'uber_lair':    './templates/navigator/navi_menu_level_uber_lair.png',
                  'uber':         './templates/navigator/navi_menu_level_uber.png',
                  'lvl1':         './templates/navigator/navi_menu_level_1.png',
                  'lvl2':         './templates/navigator/navi_menu_level_2.png',
                  'lvl3':         './templates/navigator/navi_menu_level_3.png',
                  'lvl4':         './templates/navigator/navi_menu_level_4.png',
                  'lvl5':         './templates/navigator/navi_menu_level_5.png',
                  'lvl8':         './templates/navigator/navi_menu_level_8.png',
                  'lvl9':         './templates/navigator/navi_menu_level_9.png',
                  'all':          './templates/navigator/navi_menu_level_all.png'
                  }

        list = {'farm':         './templates/navigator/navi_list_farm.png',
                'lumber':       './templates/navigator/navi_list_lumber.png',
                'iron':         './templates/navigator/navi_list_iron.png',
                'stone':        './templates/navigator/navi_list_stone.png',
                'silver':       './templates/navigator/navi_list_silver.png',
                'invader_lair': './templates/navigator/navi_list_invader_lair.png',
                'invader':      './templates/navigator/navi_list_invader.png',
                'ghost':        './templates/navigator/navi_list_ghost.png',
                'lvl1':         './templates/navigator/navi_list_lvl1.png',
                'lvl6':         './templates/navigator/navi_list_lvl6.png',
                'lvl8':         './templates/navigator/navi_list_lvl8.png',
                'lvl9':         './templates/navigator/navi_list_lvl9.png',
                'uber_lair':    './templates/navigator/navi_list_uber_lair.png',
                #'uber':         './templates/navigator/navi_menu_level_uber.png',
                #'all':          './templates/navigator/navi_menu_level_all.png'
                }

        mine_icons = {'farm':       ('./templates/navigator/mine_icon_farm6.png',),
                 'lumber':          ('./templates/navigator/mine_icon_lumber6.png',),
                 'iron':            ('./templates/navigator/mine_icon_iron6.png',),
                 'stone':           ('./templates/navigator/mine_icon_stone6.png',),
                 'silver':          ('./templates/navigator/mine_icon_silver6.png',),
                 'invader_lair':    ('./templates/navigator/mine_icon_uber_lair_1.png',
                                     './templates/navigator/mine_icon_uber_lair_2.png',),
                 }
        troops_icons = {'trebuchet': './templates/navigator/troops_icon_trebuchet.png',}

        units = {'mine_free':           './templates/navigator/mine_free.png',
                 'mine_border_left':    './templates/navigator/mine_border_left.png',
                 'mine_border_right':   './templates/navigator/mine_border_right.png',
                 'coord_yes':           './templates/navigator/b_coord_yes.png',
                 'mine_b_capture':      './templates/navigator/mine_b_capture.png',
                 'capture':             './templates/navigator/b_capture.png',
                 'troops_border':       './templates/navigator/troops_border.png',
                 'troops_field':        './templates/navigator/troops_field.png',
                 'troops_send':         './templates/navigator/troops_b_send.png',
                 'troops_checkbox':     './templates/navigator/troops_checkbox.png',
                 'troops_send2':        './templates/navigator/troops_b_send_2.png',
                 'max_marches_close':   './templates/navigator/troops_max_marches_b_close.png',
                 'troops_shield_warn':  './templates/navigator/troops_shield_warn.png',
                 'troops_shield_no':    './templates/navigator/troops_shield_no.png',

                 #'send_res':        './templates/b_market_send_res.png',
                 #'send_area_border':'./templates/border_market_send.png',
                 #'res_panel_border':'./templates/border_panel_res.png',
                 #'max_res_border_top':'./templates/border_max_res_top.png',
                 #'max_res_border_bottom':'./templates/border_max_res_bottom.png',
                 #'slider':          './templates/b_market_slider.png',
                 #'send':            './templates/b_market_send.png'
                 }

        unit_order = []

        type_select_exist = False
        level_select_exist = False
        mine_icon_exist = False
        capture_exist = False
        max_marches = False
        result = False

        current_type = ''
        current_level = ''

        icon = Icon(self.screenshot_area, icon_treshold = 0.9)
        for i_templ in icons:
            attempts = 2 if (i_templ == icons[2]) else 1
            rand_mov_chance = 1 if (i_templ == icons[1]) else 0
            icon.icon_template = i_templ
            icon_exist = icon.Find(click = True, attempts = attempts, delay = False, rand_mov_chance = rand_mov_chance)
            SleepRand(acp_short[0], acp_short[1])
        if not icon_exist:
            return False, max_marches

        navi= Window(self.screenshot_area)
        navi.action_button_treshold = 0.99
        for type, templ in types.items():
            print('%s -> %s' % (type, templ)) if debug else 0
            navi.action_button_template = templ
            type_select_exist = navi.FindUnit('ACTION', rand_mov_chance = 0, attempts = 1, delay = False)
            print('type_select_exist = %s' % type_select_exist) if debug else 0
            if type_select_exist:
                current_type = type
                break
        print('current_type: %s' % current_type)
        print('need_type: %s' % need_type)
        if current_type == need_type:
            for level, templ in levels.items():
                print('%s -> %s' % (level, templ)) if debug else 0
                navi.action_button_template = templ
                level_select_exist = navi.FindUnit('ACTION', rand_mov_chance = 0, attempts = 1, delay = False)
                print('level_select_exist = %s' % level_select_exist) if debug else 0
                if level_select_exist:
                    current_level = level
                    break
            if not level_select_exist:
                print('Can\'t find any template of level')
                return False, max_marches
            print('current_level: %s' % current_level)
            print('need_level: %s' % need_level)
            if current_level != need_level:
                unit_order.append(levels[current_level])
                unit_order.append(list[need_level])
        elif current_type != '':
            unit_order.append(types[current_type])
            unit_order.append(list[need_type])
            unit_order.append(list[need_level])
        else:
            return False, max_marches


        print(unit_order)

        for templ in unit_order:
            navi.action_button_template = templ
            navi.FindUnit('ACTION',
                          press = True,
                          #delay = ren_click_delay,
                          rand_mov_chance = 0, delay = False)

        icon_area = WindowArea(self.screenshot_area,
                         left_border_templ = units['mine_border_left'],
                         right_border_templ = units['mine_free'])
        button_area = WindowArea(self.screenshot_area,
                         left_border_templ = units['mine_free'],
                         right_border_templ = units['mine_border_right'])
        icon_area_exist = icon_area.DefineArea(direction = 'right_to_left')
        button_area_exist = button_area.DefineArea(direction = 'left_to_right')

        print('icon_area_exist: %s' % icon_area_exist)
        print('button_area_exist: %s' % button_area_exist)
        if button_area_exist:
            capture_button_exist = button_area.FindElem(action_elem_templ = units['mine_b_capture'],
                                                      attempts = 2)
        else:
            return False, max_marches
        print('capture_button_exist: %s' % capture_button_exist)
        if icon_area_exist:
            for icon_templ in mine_icons[need_type]:
                print('icon_templ: %s' % icon_templ)
                mine_icon_exist = icon_area.FindElem(action_elem_templ = icon_templ,
                                                attempts = 1)
        else:
            return False, max_marches
        count = 0
        trops_sent = False
        while not trops_sent and count < 2:
            count += 1
            capture_button_clicked = False
            if capture_button_exist and count == 1:
                capture_button_clicked = button_area.ClickElem(delay = False, rand_mov_chance = 0)
                navi.action_button_template = units['max_marches_close']
                max_marches_close_exist = navi.FindUnit('ACTION', press = True, attempts = 1)
                if max_marches_close_exist:
                    self.CloseAllWindows()
                    max_marches = True
                    return False, max_marches
            if mine_icon_exist and count == 2:
                icon_area.ClickElem(delay = False, rand_mov_chance = 0)
                navi.action_button_template = units['coord_yes']
                navi.FindUnit('ACTION', press = True, delay = False, rand_mov_chance = 0.4)
                MoveCurAndClick(scr_center['x'], scr_center['y'])
#self.slip = DefineSlip(button_pos, border = 20, field = field)
                navi.action_button_template = units['capture']
                capture_exist = navi.FindUnit('ACTION', press = True, rand_mov_chance = 0, delay = False, attempts = 2)
                navi.action_button_template = units['max_marches_close']
                max_marches_close_exist = navi.FindUnit('ACTION', press = True, attempts = 1)
                if max_marches_close_exist:
                    self.CloseAllWindows()
                    max_marches = True
                    return False, max_marches
            if capture_exist or capture_button_clicked:
                field_exist = False
                area = WindowArea(self.screenshot_area,
                                left_border_templ = troops_icons['trebuchet'],
                                right_border_templ = units['troops_border'],
                                action_elem_templ = units['troops_field'])
                area_exist = area.DefineArea(direction = 'left_to_right', scroll = 15)
                if area_exist:
                    field_exist = area.FindElem()
                if field_exist:
                    area.ClickElem(delay = False, rand_mov_chance = 0.3, field = True)
                    EnterText(number_of_troops, delay = False)
                    unit_order = [units['troops_send'],
                            units['troops_checkbox'],
                            units['troops_send2'],
                            units['troops_shield_warn'],
                            units['troops_shield_no']]
                    for templ in unit_order:
                        navi.action_button_template = templ
                        attempts = 2 if (templ == unit_order[0]) else 1
                        press = False if (templ == unit_order[3]) else True
                        unit_exist = navi.FindUnit('ACTION', press = press, rand_mov_chance = 0, attempts = attempts, delay = False)
                        if unit_exist and (templ == unit_order[0]):
                            window_exist = navi.CheckHeader(attempts = 1, lang = 'rus',
                                            h_substr = 'Атака', h_template = './templates/header_border_1.png',
                                            h_treshold = 0.5)
                            if window_exist:
                                ClickRandDelay()
                            result = True
                        if unit_exist and (templ == unit_order[3]):
                            result = False
                        if unit_exist and (templ == unit_order[4]):
                            navi.FindUnit('CLOSE', press = press, rand_mov_chance = 0, attempts = attempts, delay = False)
                trops_sent = result
            else:
                self.CloseAllWindows()
                return result, max_marches
        else:
            self.CloseAllWindows()
        return result, max_marches


#######################################################################

    def TakeTechWorks(self):
        FuncIntro(func_name = inspect.stack()[0][3])
        tw_icon = Icon(self.screenshot_area,
                       icon_template = './templates/icon_tech_works.png',
                       icon_treshold = 0.7)
        icon_exist = tw_icon.Find()
        if (icon_exist):
            print('Icon exist')
            tw_icon.Click()
            b_template = './templates/b_take_tech_works.png'
            b_treshold = 0.5
            tw_window = Window(self.screenshot_area, b_template = b_template, b_treshold = b_treshold)
            button_exist = tw_window.FindUnit('ACTION')
            if (button_exist):
                print('Кнопка найдена!')
                tw_window.PressButton('ACTION')
                print('button_exist =', button_exist)
            else:
                print('Кнопка не найдена!')

    def MultiBanUser(self, num = 1, close = True):
        FuncIntro(func_name = inspect.stack()[0][3])
        buttons = ('./templates/temp/b_ban_1.png',
                   './templates/temp/b_ban_2.png',
                   './templates/temp/b_unban_1.png',
                   './templates/temp/b_yes.png',
                   './templates/temp/b_close.png')
        usr_window = Window(self.screenshot_area,
                            b_treshold = 0.9)
        i = 0
        pressed = 0
        while (pressed != num):
            i += 1
            print('i =', i)
            for templ in buttons:
                print(templ)
                usr_window.action_button_template = templ
                button_exist = False

                # для кнопок из штатной последовательности делаем несколько попыток поиска
                # для нестандартной кнопки - только одна попытка
                attempts = 1 if (templ == buttons[4]) else 3

                button_exist = usr_window.FindUnit('ACTION',
                                                   press = True,
                                                   delay = False,
                                                   rand_mov_chance = 0.001,
                                                   attempts = attempts)
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
                close_button_exist = usr_window.FindUnit('CLOSE', press = True, rand_mov_chance = 0.1)
                RandomMouseMove(treshold = 1)
                print(close_button_exist)

    def RenameAcc(self, start = 1, end = 2):
        FuncIntro(func_name = inspect.stack()[0][3],
                  params = {'start':start, 'end': end},
                  only_print = False)
        icons = ('./templates/tab_panel_1.png',
                 './templates/icon_panel_menu.png',
                 './templates/menu_skins.png',
                 './templates/tab_palace_profile.png')
        buttons = ('./templates/temp/b_edit.png',
                   './templates/temp/field_1.png',
                   './templates/temp/b_apply.png')

        icon = Icon(self.screenshot_area)
        for i_templ in icons:
            attempts = 1 if (i_templ == icons[0]) else 2
            icon.icon_template = i_templ
            icon.Find(click = True, attempts = attempts)

        profile_window = Window(self.screenshot_area,
                                b_treshold = 0.9)
        current_number = start-1
        pressed = 0
        good = True
        while (current_number != end):
            if good:
                current_number += 1
            else:
                started = self.StartGame(from_desktop = False)
                if started:
                    self.RenameAcc(start = current_number, end = end)
            print('current_number =', current_number)
            cnt = 0
            good = False
            for templ in buttons:

                profile_window.action_button_template = templ
                if (templ == buttons[1]):
                    profile_window.action_button_treshold = 0.99
                    is_field = True
                else:
                    profile_window.action_button_treshold = 0.9
                    is_field = False
                button_exist = profile_window.FindUnit('ACTION',
                                                       press = True,
                                                       delay = ren_click_delay,
                                                       rand_mov_chance = 0.001,
                                                       field = is_field)
                if (templ == buttons[0] and button_exist):
                    cnt += 1
                if (templ == buttons[1] and button_exist):
                    cnt += 1
                if (templ == buttons[2] and button_exist):
                    if cnt == 2:
                        pressed += 1
                        good = True
                    print ('pressed =', pressed)
                    print ('current_number =', current_number)

                if (current_number > 1 and cnt == 2):
                    j = 0
                    while (j < len(str(current_number-1))):
                        j += 1
                        pg.hotkey('backspace')
                        SleepRand(0.1, 0.5)

                if (templ == buttons[1] and cnt == 2):
                    SleepRand(acp_short[0], acp_short[1])
                    EnterText(text = str(current_number))

        print('*****************')
        print('TOTAL RENAMES: %s' %pressed)
        print('End time: %s' %NowToStr())
        print('*****************')
        log_text = 'TOTAL RENAMES: %s' %pressed
        WriteLog(log_text)


    def KickFromValley(self, num = 1,
                       city_name = 'RockCity',
                       lang = 'eng',
                       coord = {'x': '3', 'y': '7'},
                       close = False,
                       suspend = False):
        FuncIntro(func_name = inspect.stack()[0][3])
        city_name = city_name.upper()
        count = 0
        buttons = ('./templates/b_to_expel.png',
                   './templates/b_to_expel_2.png',
                   './templates/b_to_expel_store.png',
                   './templates/b_to_expel_close.png',
                   './templates/b_to_expel_close2.png',
                   )
        pressed = 0
        while (pressed < num) and (count < num*3):
            count += 1
            print('count = %s' %count)
            MoveCurAndClick(scr_center['x'], scr_center['y'])
            wind = Window(self.screenshot_area,
                          h_substr = city_name,
                          h_template = './templates/header_border_2.png',
                          b_treshold = 0.8)
            promo_wind = Window(self.screenshot_area,
                                h_substr = 'годное',
                                h_template = './templates/header_border_2.png',
                                b_treshold = 0.8)
            header_exist = False
            promo_exist = False
            check_count = 0
            while (not header_exist) and (check_count < 2000):
                check_count += 1
                print('check_count: %s' %check_count)
                header_exist = wind.CheckHeader(lang = lang, attempts = 1)
                if header_exist:
                    break
                wind.close_button_treshold = 0.9
                promo_window_exist = promo_wind.CheckHeader(lang = 'rus', attempts = 1)
                close_b_exist = wind.FindUnit('CLOSE', press = False)
                print('close_b_exist = {}'.format(close_b_exist))
                if (not close_b_exist) or promo_window_exist:
                    game_is_started = self.StartGame(from_desktop = False)
                    map_is_opened = self.GoToMap()
                    SleepRand(acp_long[0], acp_long[1])
                    self.JumpTo(times = 1, coord = coord, no_move = True)
                    MoveCurAndClick(scr_center['x'], scr_center['y'])
                SleepRand(acp_short[0], acp_short[1])

            if header_exist:
                for templ in buttons:
                    attempts = 1 if ((templ == buttons[2]) or (templ == buttons[3]) or (templ == buttons[4])) else 3
                    wind.action_button_template = templ
                    button_exist = wind.FindUnit('ACTION', press = True, delay = False, attempts = attempts, rand_mov_chance = 0)
                    if (templ == buttons[2]) and button_exist:
                        print('No seal! You must buy it.')
                        return False
                    if (templ == buttons[1]) and button_exist:
                        pressed += 1
                        print('pressed = %s' %pressed)
                    if (templ == buttons[3] or templ == buttons[4]) and button_exist:
                        pressed -= 1
                        print('pressed = %s' %pressed)
            else:
                print('No more time!')
                return False
            SleepRand(acp_long[0], acp_long[1])
        print('*****************')
        print('TOTAL KICKS: %s' %pressed)
        print('End time: %s' %time.asctime())
        print('*****************')
        if close:
            self.CloseGame()
        if suspend:
            SuspendSystem()
        return True


    def JumpTo(self, times = 1, coord = {'x': '2', 'y': '6'}, no_move = False):
        FuncIntro(func_name = inspect.stack()[0][3])
        coord_units = ('./templates/map_coord_x.png',
                       './templates/map_coord_y.png',
                       './templates/map_coord_go.png',

                       )
        move_units = ('./templates/b_map_town_look.png',
                      './templates/b_map_move_apply.png',
                      './templates/b_map_move_yes.png')
        success = 0
        count = 0
        while (success < times) and (count < times*2):
            goal_icon = Icon(self.screenshot_area,
                             icon_template = './templates/map_coord.png',
                             icon_treshold = 0.9)
            window = Window(self.screenshot_area)
            goal_icon_exist = goal_icon.Find(click = True, delay = False)
            if goal_icon_exist:
                step = 0
                for templ in coord_units:
                    field = False if (templ == coord_units[2]) else True
                    window.action_button_template = templ
                    unit_exist = window.FindUnit('ACTION',
                                                 press = True,
                                                 field = field,
                                                 rand_mov_chance = 0,
                                                 delay = False)
                    if not unit_exist:
                        print('Unit not exist. Time: %s' %time.asctime())
                        print('Unit: %s' %templ)
                        CutThePict(self.screenshot_area, png=True)
                        continue

                    step += 1 if unit_exist else 0
                    if (templ == coord_units[0]) and (step == 1):
                        EnterText(text = coord['x'])
                    if (templ == coord_units[1]) and (step == 2):
                        EnterText(text = coord['y'])
            else:
                print('Goal not exist. Time: %s' %time.asctime())
                CutThePict(self.screenshot_area, png=True)
                self.CloseAllWindows()
                game_is_started = self.StartGame(from_desktop = False)
                map_is_opened = self.GoToMap()
                continue
            if not no_move:
                MoveCurAndClick(scr_center['x'], scr_center['y'])
                SleepRand(acp_short[0], acp_short[1])
                for templ in move_units:
                    window.action_button_template = templ
                    if (templ == move_units[0]):
                        press = False
                        attempts = 1
                    else:
                        press = True
                        attempts = 3
                    button_exist = window.FindUnit('ACTION', press = press, delay = False, attempts = attempts)
                    if (templ == move_units[0]) and button_exist:
                        self.CloseAllWindows()
                        break
                #button_exist = True
                if unit_exist and button_exist:
                    success += 1
            else:
                if unit_exist:
                    success += 1

            #if times > 1:
            #    SleepRand(25, 30)
        if success == times:
            return True
        else:
            return False


    def GoToMap(self):
        FuncIntro(func_name = inspect.stack()[0][3])
        icons = ('./templates/tab_panel_1.png',
                 './templates/icon_panel_map.png'
                 )

        icon = Icon(self.screenshot_area, icon_treshold = 0.9)
        for i_templ in icons:
            attempts = 2 if (i_templ == icons[1]) else 1
            rand_mov_chance = 1 if (i_templ == icons[1]) else 0
            icon.icon_template = i_templ
            icon_exist = icon.Find(click = True, attempts = attempts, delay = False, rand_mov_chance = rand_mov_chance)
            SleepRand(acp_short[0], acp_short[1])
        #map_icon = Icon(self.screenshot_area,
        #                icon_template = './templates/icon_panel_map.png',
        #                icon_treshold = 0.9)
        #map_icon_exist = map_icon.Find(click = True)
        return icon_exist


    def TakeEventMovings(self, attempt = 1):
        FuncIntro(func_name = inspect.stack()[0][3])
        news_event_labels = {
                             'cvc':   './templates/events/label_news_cvc.png',
                             'kvk':   './templates/events/label_news_kvk.png',
                             'kvk_r': './templates/events/label_news_kvk_revenge.png',
                             'kvk_f': './templates/events/label_news_kvk_fury.png',
                             't_battle': './templates/events/label_news_throne_battle.png',
                             'invader': './templates/events/label_news_invader.png',
                            }
        event_icons = {
                       'cvc':   './templates/events/icon_cvc.png',
                       'kvk':   './templates/events/icon_kvk.png',
                       'kvk_r': './templates/events/icon_kvk_revenge.png',
                       'kvk_f': './templates/events/icon_kvk_fury.png',
                       't_battle': './templates/events/icon_throne_battle.png',
                      }
        units = {
                 'icon_news':           './templates/events/icon_news.png',
                 'icon_events':         './templates/events/icon_events.png',
                 'b_read_all':          './templates/events/b_read_all.png',
                 #'b_take':              './templates/events/b_take.png',
                 'label_expected':      './templates/events/label_expected.png',
                 'border_right_event':  './templates/events/border_right_event.png',
                }
        take_units = {
                      'b_take':              './templates/events/b_take.png',
                      'b_take_1':            './templates/events/b_take_1.png',
        }
        news_icon = Icon(self.screenshot_area,
                    icon_template = units['icon_news'],
                    icon_treshold = 0.9)
        events_icon = Icon(self.screenshot_area,
                           icon_template = units['icon_events'],
                           icon_treshold = 0.9)
        news_icon_exist = news_icon.Find(click = True, delay = True, rand_mov_chance = 0)
        news_window = Window(self.screenshot_area,
                             b_template = units['b_read_all'],
                             h_substr = 'сегодня',
                             h_template = './templates/header_border_2.png',
                             b_treshold = 0.8,
                             e_treshold = 0.8)
        curr_event = 'no_event'
        if news_icon_exist:
            upper_key = 'no_event'
            upper_event_label_top = 2000
            for key, value in news_event_labels.items():
                #print('key: {}, value: {}'.format(key, value))
                news_window.element_template = value
                event_label_exist = news_window.CheckElement(fresh_screen = True)
                if event_label_exist:
                    event_label_position = news_window.GetElementPos()
                    event_label_top = event_label_position['top_left'][1]
                    print('event_label_top = {}'.format(event_label_top))
                    print('upper_event_label_top = {}'.format(upper_event_label_top))
                    # вычисляем последнюю новость по относительной высоте идентифицирующего элемента
                    # если текущий элемент выше самого высокого, считаем, что он самый высокий на текущий моемент
                    if event_label_top < upper_event_label_top:
                        upper_event_label_top = event_label_top
                        upper_key = key
            if upper_key not in ('invader'):
                curr_event = upper_key
            print('curr_event = {}'.format(curr_event))
            take_exist = False
            if curr_event != 'no_event':
                self.CloseAllWindows()
                events_icon_exist = events_icon.Find(click = True, delay = True, rand_mov_chance = 0)
                events_window = Window(self.screenshot_area,
                                       b_template = event_icons[curr_event],
                                       b_treshold = 0.9)

                events_window.action_button_template = units['border_right_event']
                elem_exist = events_window.FindUnit('ACTION', move_to = True, delay = False, rand_mov_chance = 0)
                print('elem_exist = {}'.format(elem_exist))

                event_exist = False

                scrolls = 0
                target_scrolls = 5

                while not event_exist and scrolls < target_scrolls:
                    events_window.action_button_template = event_icons[curr_event]
                    event_exist =  events_window.FindUnit('ACTION', press = True, delay = True, rand_mov_chance = 0, attempts = 1)
                    if not event_exist:
                        ScrollMouse(number = 5, direction = 'DOWN')
                        scrolls += 1
                if event_exist:
                    #events_window.action_button_template = take_units['b_take']
                    count = 0
                    while not take_exist and attempt < 3:
                        for key, value in take_units.items():
                            events_window.action_button_template = value
                            take_exist = events_window.FindUnit('ACTION', press = True, delay = False, rand_mov_chance = 0, attempts = 2)
                        if not take_exist:
                            self.StartGame(from_desktop = False)
                            self.TakeEventMovings(attempt = attempt + 1)
                            return 0
                        else:
                            self.CloseAllWindows()
                else:
                    self.StartGame(from_desktop = False)
                    self.TakeEventMovings()
                    return 0

            attempts = 0
            if take_exist or (not take_exist and attempt == 3) or (curr_event == 'no_event'):
                news_icon_exist = False
                count = 0
                while not news_icon_exist and count < 5:
                    news_icon_exist = news_icon.Find(click = True, delay = True)
                    news_window_exist = news_window.CheckHeader(lang = 'rus', attempts = 1)
                    if news_icon_exist or news_window_exist:
                        news_window.FindUnit('ACTION', press = True, delay = False, rand_mov_chance = 0)
                        self.CloseAllWindows()
                    else:
                        count += 1
                        self.StartGame(from_desktop = False)








##################################################

def DefineSlip(position, border = 20, field = False):
    slip = {'x': 0, 'y': 0}
    templ_width = position['bottom_right'][0] - position['top_left'][0]
    templ_height = position['bottom_right'][1] - position['top_left'][1]
    border_x = round((templ_width/100)*20)
    border_y = round((templ_height/100)*20)
    slip['x'] = round(templ_width/2)-border_x
    slip['y'] = round(templ_height/2)-border_y if not field else 3
    return slip


def TakeManyTasks(acc = 'COOLROCK2', take_tab = 2, take_number = 2, suspend = False, close = False):
    FuncIntro(func_name = inspect.stack()[0][3],
              params = {'acc':acc, 'take_tab': take_tab, 'take_number': take_number, 'suspend': suspend, 'close': close},
              only_print = False)

    player = Vik_acc(account_name = acc)
    started = player.StartGame(from_desktop = True)
    if started:
        player.TakeTasks(take_tab = take_tab, take_number = take_number)
    if close:
        player.CloseGame()
    if suspend:
        SuspendSystem()





def KickInTwoWind(times = 1, city_name = 'RockCity', lang = 'eng', coord = {'x': '3', 'y': '7'}, suspend = False):
    print('******************************************')
    print('******************************************')
    FuncIntro(func_name = inspect.stack()[0][3],
              params = {'times': times, 'city_name': city_name, 'suspend': suspend},
              only_print = False)

    player = Vik_acc(account_name = 'coolrock2')
    count = 0
    success = 0
    while (success < times) and (count < times*2):
        count += 1
        jump = player.JumpTo(times = 1, coord = coord)
        if not jump:
            print('Bad result of JumpTo')
            break
        print('Switch to Temperok')
        SwitchWindow()
        #RockCity
        kick = player.KickFromValley(num = 1, city_name = city_name, lang = lang, coord = coord)
        if not kick:
            print('Bad result of KickFromValley')
            break
        print('Switch to CoolRock')
        SwitchWindow()
        if jump and kick:
            success += 1
            print('success: %s' %success)
    print('*****************')
    print('TOTAL SUCCESS: %s' %success)
    print('End time: %s' %time.asctime())
    print('*****************')
    if suspend:
        SuspendSystem()


def RenameAccount(acc = 'COOLROCK2', start = 356, end = 365, suspend = False, close = False):
    FuncIntro(func_name = inspect.stack()[0][3],
              params = {'acc':acc, 'start': start, 'end': end, 'suspend': suspend, 'close': close},
              only_print = False)

    player = Vik_acc(account_name = acc)
    started = player.StartGame(from_desktop = True)
    if started:
        player.RenameAcc(start = start, end = end)
    if close:
        player.CloseGame()
    if suspend:
        SuspendSystem()


def ProccessAllAccounts(suspend = False, with_pause = False, short = 0):
    FuncIntro(func_name = inspect.stack()[0][3], params = {'suspend': suspend}, only_print = False)

    for acc in accounts:
        print(acc)
        not_send = (
                    'KRAKOZ',
                    'TEMPER'
                    )
        not_loki = (
                    'DUBROVSK',
                    'KRAKOZ',
                    'TEMPER'
                    )
        mining_days = (1, 4)
        mining_accs = (
                       #'COOLROCK2',
                       'FARAMEAR',
                       'COOLROCK4',
                       #'OXYIRON',
                       'TRUVOR',

        )
        number_of_troops = 3000 if acc == 'TRUVOR' else 10000
        #if acc in (
        #           'OXYIRON',
        #           'COOLROCK2',
        #           'COOLROCK4',
        #           'FARAMEAR',
        #           'DUBROVSK',
        #           'TRUVOR',
        #           'KRAKOZ',
        #           ):
        #    continue
        if short == 1 and acc in ('OXYIRON',
                             'COOLROCK2',
                             'COOLROCK4',
                             #'FARAMEAR',
                             'DUBROVSK',
                             'TRUVOR',
                             'KRAKOZ',
                             'TEMPER'
                   ):
            continue
        if short == 2 and acc in (#'OXYIRON',
                             #'COOLROCK2',
                             #'COOLROCK4',
                             #'FARAMEAR',
                             'DUBROVSK',
                             'TRUVOR',
                             'KRAKOZ',
                             'TEMPER'
                   ):
            continue
        day_of_week = dttm.datetime.today().isoweekday()
        player = Vik_acc(account_name = acc)
        started = player.StartGame(from_desktop = True)
        if started:
            player.CloseAllWindows()
            if acc not in not_loki:
                player.TakeChestOfLoki()
            player.TakeTechWorks()
            player.TakeEventMovings()
            player.TakeQuests()
            count = 0
            while count < 2:
                count += 1
                player.TakeTasks(take_tab = 3, take_number = 1)
                player.TakeQuests()
                player.PressHelp()
                player.TakeDailyLoyalBonus()
                player.TakeEveryDayBank()
                if acc not in not_loki:
                    player.TakeChestOfLoki()
            send_attempts = 3
            send_count = 0
            res_sent = False
            while not res_sent and (send_count < send_attempts) and (acc not in not_send):
                send_count += 1
                print('send_count = {}'.format(send_count))
                res_sent = player.SendResources(res_list = [])
            if acc not in not_loki:
                player.TakeChestOfLoki()
            player.TakeQuests()
            #if (day_of_week in mining_days) and (acc in mining_accs):
            #    CollectRes(acc = acc,
            #               need_type = accounts[acc]['default_mine'],
            #               need_level = 'lvl6',
            #               number_of_troops = number_of_troops,
            #               times = 12,
            #               to_the_max_marches = True,
            #               close = False,
            #               suspend = False)

            if with_pause:
                CountDown(seconds = 50)
            player.CloseGame()
            SleepRand(1, 2)
        else:
            break
    if suspend:
        SuspendSystem()


def ResourceCollectMode(priod_in_minuts = 60, disp = 5):
    priod_in_minuts = random.randint(priod_in_minuts-disp, priod_in_minuts+disp)
    suspend_duration = priod_in_minuts*60
    short = 0
    count = 1
    while count >= 0:
        if count > 2:
            SleepRand(10, 15)
            RandomMouseMove(treshold = 1)

        if (count % 2) == 0:
            short = 1
        elif (count % 3) == 0:
            short = 2
        else:
            short = 0
        count += 1

        ProccessAllAccounts(short = short)
        SuspendSystem(sleep_time = 120, duration = suspend_duration)



def CollectRes(acc = 'COOLROCK2',
               need_type = 'farm',
               need_level = 'lvl6',
               number_of_troops = '1111',
               times = 5,
               to_the_max_marches = False,
               close = False,
               suspend = False):
    FuncIntro(func_name = inspect.stack()[0][3], params = {'suspend': suspend}, only_print = False)
    #default_sleep_time = 60
    default_sleep_time = round(int(number_of_troops)/100)
    current_sleep_time = default_sleep_time
    count = 0
    sent = 0
    max_marches_count = 0
    player = Vik_acc(account_name = acc)
    started = player.StartGame(from_desktop = True)
    if started:
        while (sent < times) and started and (count < 2*times):
            print('sent: %s, count: %s, max_marches_count: %s' % (sent, count, max_marches_count))
            count += 1
            success, max_marches = player.SendToMine(need_type = need_type, need_level = need_level, number_of_troops = number_of_troops)
            print('-----')
            print('success, max_marches: %s, %s' % (success, max_marches))
            if success:
                sent += 1
                count = sent
                max_marches_count = 0
                current_sleep_time = default_sleep_time
            elif not success and not max_marches:
                if (count - sent) > 5:
                    player.CloseGame()
                started = player.StartGame(from_desktop = False)
            elif not success and max_marches:
                if not to_the_max_marches:
                    max_marches_count += 1
                    CountDown(seconds = current_sleep_time)
                    current_sleep_time *= 2
                else:
                    break
                #    return True
    print('*********************')
    print('TOTAL sent: %s, count: %s, max_marches_count: %s' % (sent, count, max_marches_count))
    print('*********************')
    if close:
        player.CloseGame()
    if suspend:
        SuspendSystem()
    if (sent == times) or (not success and max_marches):
        return True
    else:
        return False


def CollectResInTwoWind(accs = ('COOLROCK2', 'FARAMEAR'),
                        need_level = 'lvl6',
                        number_of_troops = '2000',
                        times = 10,
                        close = False,
                        suspend = False):
    success = 0
    while success < times:
        count = 0
        for acc in accs:
            count += 1
            result = CollectRes(acc = acc,
                            need_type = accounts[acc]['default_mine'],
                            need_level = need_level,
                            number_of_troops = number_of_troops,
                            times = 12,
                            to_the_max_marches = True,
                            close = False,
                            suspend = False)
            SwitchWindow()
            if (count == len(accs)) and result:
                success += 1


def CollectResQueue(accs = ('COOLROCK2', 'FARAMEAR'),
                        need_level = 'lvl6',
                        number_of_troops = '15000',
                        times = 20,
                        close = False,
                        suspend = False):
    success = 0
    while success < times:
        count = 0
        for acc in accs:
            count += 1
            result = CollectRes(acc = acc,
                            need_type = accounts[acc]['default_mine'],
                            need_level = need_level,
                            number_of_troops = number_of_troops,
                            times = accounts[acc]['default_marches'],
                            to_the_max_marches = True,
                            close = True,
                            suspend = False)
            #SwitchWindow()
            if (count == len(accs)) and result:
                success += 1
                print("SUCCESS: {}".format(success))
                if success%3 == 0:
                    ProccessAllAccounts(suspend = False)


def CountDown(seconds = 120):
    FuncIntro(func_name = inspect.stack()[0][3], params = {'seconds':seconds}, only_print = True)

    print('Countdown - %s seconds...' %seconds)
    values = range(seconds, 0, -1)
    for i in values:
        print(' => %02d <=' %i, end='\r')
        time.sleep(1)
    RandomMouseMove(treshold = 1)
    #print ("\n\r", end="")


Init()
#coolrock = Vik_acc(account_name = 'COOLROCK2')
coolrock = Vik_acc(account_name = 'COOLROCK4')
#coolrock = Vik_acc(account_name = 'OXYIRON')
#9 - ResourceCollectMode()
#10 - ProccessAllAccounts()
#14 - multi CollectRes
#15 - CollectResQueue
r = 9
###coolrock.TakeEventMovings()
#ProccessAllAccounts(suspend = False)
#suspend_duration = 40
#SuspendSystem(sleep_time = 120, duration = suspend_duration)
#ProccessAllAccounts(suspend = False, short = 2)
#coolrock.TakeTasks(take_tab = 2, take_number = 150, start_game = True, close_game = True)
suspend_duration = 340*60
suspend_duration = 4*60*60
suspend_duration = 7*60*60
#suspend_duration = 1*60*60
#SuspendSystem(sleep_time = 120, duration = suspend_duration)
#ProccessAllAccounts(suspend = False)


SleepRand(acp_long[0], acp_long[1])
if r == 0:
    #SuspendSystem(sleep_time = 9, duration = 10)
    ResourceCollectMode(priod_in_minuts = 1, disp = 0)
    #coolrock.SendToMine()
if (r == 1) or (r == 6):# or (r == 14):
    coord = {'x': '3', 'y': '7'}
    #coord = {'x': '2', 'y': '7'}
    #coolrock.MultiBanUser(72, close = False)
    #coolrock.RenameAcc(start = 889, end = 410)
    #coolrock.KickFromValley(num = 10, city_name = 'CoRockCity', lang = 'eng', suspend = True)
    #coolrock.KickFromValley(num = 648, city_name = 'lionkingdom', lang = 'eng', suspend = True)
    #coolrock.KickFromValley(num = 10, city_name = 'Tyumen', lang = 'eng', suspend = False)
    #coolrock.KickFromValley(num = 148, city_name = 'Боевая', lang = 'rus', suspend = False, coord = coord)
    #coolrock.KickFromValley(num = 357, city_name = 'ИВАНОВО', lang = 'rus')
    #coolrock.KickFromValley(num = 10, city_name = 'Dust', lang = 'eng')
    #coolrock.KickFromValley(num = 10, city_name = 'dragons', lang = 'eng')
    #coolrock.KickFromValley(num = 44, city_name = 'Dorn', lang = 'eng', close = True)
    coolrock.KickFromValley(num = 220, city_name = 'city_4684', lang = 'eng', coord = coord)
    #coolrock.KickFromValley(num = 1, city_name = 'драгунка', lang = 'rus')
    #coolrock.TakeTasks(take_tab = 3, take_number = 200)
    ProccessAllAccounts(suspend = False)

if (r == 2) or (r == 4) or (r == 6):
    #coolrock.StartGame(from_desktop = True)
    coolrock.CloseAllWindows()
    coolrock.TakeDailyLoyalBonus()
    #coolrock.TakeChestOfLoki()
    coolrock.PressHelp()
    coolrock.TakeEveryDayBank()
    coolrock.TakeTasks(take_tab = 3, take_number = 1)
    coolrock.TakeTechWorks()
    coolrock.TakeQuests()
if (r == 3) or (r == 4):
    #coolrock.SendResources(receiver_name = r'CoolRock-4')
    coolrock.SendResources()

    #coolrock.SendResources(receiver_name = r'Цита')
if r == 5:
    coolrock.PressHelp(help_mode = True)

if r == 7:
    coolrock.CloseGame()

if r == 9:
    ResourceCollectMode(priod_in_minuts = 50, disp = 10)
if r == 10:
    suspend = False
    with_pause = False
    #suspend = True
    with_pause = True
    #CountDown(seconds = 2000)
    ProccessAllAccounts(suspend = suspend, with_pause = with_pause)
    #TakeManyTasks(acc = 'COOLROCK2',
    #              take_tab = 3,
    #              take_number = 23,
    #              close = True,
    #              suspend = False)

if r == 11:
    #ProccessAllAccounts(suspend = False)
    RenameAccount(acc = 'COOLROCK2', start = 951, end = 1000, suspend = False, close = True)
    #RenameAccount(acc = 'COOLROCK4', start = 351, end = 650, suspend = False, close = True)
    ProccessAllAccounts(suspend = True)
    #RenameAccount(acc = 'OXYIRON', start = 226, end = 350, suspend = False, close = True)
    #RenameAccount(acc = 'FARAMEAR', start = 26, end = 50, suspend = False, close = True)
    #RenameAccount(acc = 'DUBROVSK', start = 351, end = 650, suspend = False, close = True)
    #ProccessAllAccounts(suspend = False)

if r == 12:
    coords = {'x': '3', 'y': '7'}
    #KickInTwoWind(times = 10, city_name = 'city_5531', lang = 'eng', coord = {'x': '252', 'y': '449'}, suspend = False)
    #KickInTwoWind(times = 10, city_name = 'city_5533', lang = 'eng', coord = coords, suspend = False)
    KickInTwoWind(times = 91, city_name = 'CoRockCity', lang = 'eng', coord = coords, suspend = False)
    #KickInTwoWind(times = 60, city_name = 'xyLand', lang = 'eng', coord = coords, suspend = False)
    #KickInTwoWind(times = 10, city_name = 'Osgiliath', lang = 'eng', coord = coords, suspend = False)
    #KickInTwoWind(times = 10, city_name = 'Город без щита', lang = 'rus', coord = coords, suspend = False)
    #KickInTwoWind(times = 10, city_name = 'Деревня', lang = 'rus', coord = coords, suspend = False)
    ##KickInTwoWind(times = 70, city_name = 'Боевая', lang = 'rus', coord = coords, suspend = False)
    #coolrock.TakeQuests()
    #CollectResInTwoWind(accs = ('COOLROCK2', 'FARAMEAR'),
    #                        need_level = 'lvl6',
    #                        number_of_troops = '8888',
    #                        times = 300,
    #                        close = False,
    #                        suspend = False)

if r == 13:
#need_type = 'invader_lair',
#need_level = 'uber_lair',
#need_type = 'farm',
#need_level = 'lvl6',
    #coolrock.SendToMine()
    ProccessAllAccounts(suspend = False)
    #CollectRes(acc = 'oxyiron',
    #               need_type = 'farm',
    #               need_level = 'lvl6',
    #               number_of_troops = '1200',
    #               times = 10,
    #               close = True,
    #               suspend = False)
    #CollectRes(acc = 'COOLROCK4',
    #               need_type = 'farm',
    #               need_level = 'lvl6',
    #               number_of_troops = '1200',
    #               times = 50,
    #               close = True,
    #               suspend = False)
    CollectRes(acc = 'COOLROCK2',
                   need_type = 'invader_lair',
                   need_level = 'uber_lair',
                   number_of_troops = '2000',
                   times = 200,
                   close = True,
                   suspend = False)
    ProccessAllAccounts(suspend = False)
    CollectRes(acc = 'COOLROCK2',
                   need_type = 'farm',
                   need_level = 'lvl6',
                   number_of_troops = '1200',
                   times = 500,
                   close = True,
                   suspend = False)
    #CollectRes(acc = 'TEMPER',
    #               need_type = 'farm',
    #               need_level = 'lvl6',
    #               number_of_troops = '4444',
    #               times = 100,
    #               close = True,
    #               suspend = False)
    ProccessAllAccounts(suspend = True)
if r == 14:
    need = 120
    count = 0
    while count < need:
        count += 1
        log_text = '>>>>>count: {}<<<<<'.format(count)
        print(log_text)
        WriteLog(log_text)
        #ProccessAllAccounts(suspend = False)
        #CollectRes(acc = 'COOLROCK2',
        #               need_type = 'invader_lair',
        #               need_level = 'uber_lair',
        #               number_of_troops = '1500',
        #               times = 300,
        #               close = True,
        #               suspend = False)

        CollectRes(acc = 'COOLROCK2',
                       need_type = 'farm',
                       need_level = 'lvl6',
                       number_of_troops = '2333',
                       times = 90,
                       close = True,
                       suspend = False)
        ProccessAllAccounts(suspend = False)
    ResourceCollectMode(priod_in_minuts = 40, disp = 10)
    #ProccessAllAccounts(suspend = True)


if r == 15:
    ProccessAllAccounts(suspend = False, short = 1)
    TakeManyTasks(acc = 'COOLROCK2',
                  take_tab = 2,
                  take_number = 100,
                  close = True,
                  suspend = False)
    ProccessAllAccounts(suspend = False, short = 1)
    CollectResQueue(accs = ('COOLROCK2',
                            'FARAMEAR',
                            #'COOLROCK4',
                            'COOLROCK2',
                            'OXYIRON',
                            'TRUVOR'
                            ),
                            need_level = 'lvl6',
                            number_of_troops = '6500',
                            times = 100,
                            close = False,
                            suspend = False)
    ResourceCollectMode(priod_in_minuts = 50, disp = 10)

#recognize_image()
#file_name = CutThePict({'top': 0, 'left': 0, 'width': scr_resolution['x'], 'height': scr_resolution['y']}, png=True)
#file_name = CutThePict({'top': 247, 'left': 830, 'width': 500, 'height': 42}, png=True)
#scr_s = CutThePict({'top': 0, 'left': 0, 'width': scr_resolution['x'], 'height': scr_resolution['y']})
#print(scr_s)
