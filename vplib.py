
import random
import time
import datetime as dttm
from datetime import datetime, date
import pyautogui as pg
import numpy as np
import cv2
from mss.linux import MSS as mss
from mss.tools import to_png
from PIL import Image, ImageDraw, ImageChops
import pytesseract
import inspect
import os

global acp_short, acp_long, lwp, debug, wc_time, load_time_treshold, scr_resolution
global accounts, acc_icons, ren_click_delay, scr_center, chrome_path, chrome_key1, chrome_key2

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
Init()
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
