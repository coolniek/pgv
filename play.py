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

wc_time = 3000
scr_resolution = {'x': 1920, 'y': 1080}


def MoveCursorRand(dest_x, dest_y, dur = 0.7):
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
        x_steps = round(x_diff/x_avg_step)
        y_avg_step = round(y_diff/x_steps)
        y_min_step, y_max_step = abs(round((y_avg_step/3)*2)), abs(round((y_avg_step/3)*5))
        if ((y_min_step <= 1) or (y_max_step <= 1)):
            y_min_step, y_max_step = 2, 8
    else:
        y_avg_step = (y_min_step + y_max_step)/2
        y_steps = round(y_diff/y_avg_step)
        x_avg_step = round(x_diff/y_steps)
        x_min_step, x_max_step = abs(round((x_avg_step/3)*2)), abs(round((x_avg_step/3)*5))
        if ((x_min_step == 0) or (x_max_step ==0)):
            x_min_step, x_max_step = 2, 8

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


def CutThePict(area, png=False):
    area = area
    sct = mss()
    img = sct.grab(area)
    if (png == False):
        img = np.array(img)
        return img
    else:
        img_name = str(time.time())[:10]+str(time.time())[-2:]
        output = img_name+".png"
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

        #cv2.rectangle(full_img, top_left, bottom_right, (0, 255, 0), 1)
        #cv2.circle(full_img, center_point, 1, (0, 255, 0), 2)
        #cv2.imshow("img", full_img)
        #cv2.waitKey(3000)
        #cv2.destroyAllWindows()

        position = {'top_left': top_left, 'bottom_right': bottom_right, 'center_point': center_point}
        found = True
    return found, position


def Init():
    global scr_resolut, wcen
    #resolution and window center
    #scr_resolut = {'x': '1280', 'y': '960'}
    #scr_resolut = {'x': 1920, 'y': 1080}
    wcen = {'x': 585, 'y': 521}


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(wc_time)
    cv2.destroyAllWindows()


def RandomForMove(x, y):
    random.seed(version=2)
    tm_default = 0.5
    params = {'x': x, 'y': y, 'tm': 0.5}
    min_x = x-5
    max_x = x+5
    min_y = y-5
    max_y = y+5
    params['x'] = random.randint(min_x, max_x)
    params['y'] = random.randint(min_y, max_y)
    params['tm'] = random.uniform(0.67, 0.9)
    return params


def MoveCurAndClick(x=0, y=0):
    sleep_min = 3
    sleep_max = 10
    print('x='+str(x)+', y='+str(y))
    mv = RandomForMove(x, y)
    print(mv)
    #pg.moveTo(mv['x'], mv['y'], mv['tm'], pg.easeOutQuad)
    MoveCursorRand(mv['x'], mv['y'], mv['tm'])
    time.sleep(random.random() + 0.4)
    pg.click()
    time.sleep(random.randint(sleep_min, sleep_max))


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
    #rec_area = {'top': wcen['y']-293, 'left': wcen['x']-165, 'width': 325, 'height': 39}
    img = CutThePict(rec_area)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = grayscale_3_levels(img)
    cv2.imshow('image', img)
    cv2.waitKey(wc_time)
    cv2.destroyAllWindows()
    text='test'
    text = pytesseract.image_to_string(img, lang="rus")
    print(text)
    return text

def ColorExist(icon_area, color_range):
    #icon_area = {'top': wcen['y']+343, 'left': wcen['x']+48, 'width': 66, 'height': 61}
    low_col = color_range['low_col']
    high_col = color_range['high_col']
    img = CutThePict(icon_area)
    #img = cv2.imread('daily_icon.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    viewImage(img)
    print(type(img), img.shape)
    print(low_col, high_col)

    col_range = cv2.inRange(img, low_col, high_col)

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

    def __init__(self, h_substr, shot_area, h_template, b_template, h_treshold, b_treshold):
        self.header_substr = h_substr
        self.shot_area = shot_area
        self.header_template = h_template
        self.action_button_template = b_template
        self.header_treshold = h_treshold
        self.action_button_treshold = b_treshold
        self.close_button_template = './templates/b_close.png'
        self.close_button_treshold = 0.99
        self.screen = CutThePict(self.shot_area)
        self.action_button_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}
        self.close_button_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}

    def CheckHeader(self):
        header_exist, header_border_pos = FindObject(self.screen, self.header_template, self.header_treshold)
        n_top = header_border_pos['top_left'][1]
        n_right = header_border_pos['bottom_right'][0]
        n_height = (header_border_pos['bottom_right'][1])-(header_border_pos['top_left'][1])
        if (header_exist):
            header_area = {'top': n_top, 'left': n_right, 'width': 500, 'height': n_height}
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

    def FindButton(self):
        button_exist, self.action_button_pos = FindObject(self.screen, self.action_button_template, self.action_button_treshold)
        return button_exist

    def PressButton(self):
        MoveCurAndClick(self.action_button_pos['center_point'][0], self.action_button_pos['center_point'][1])


class Icon(object):
#shot_area, ico_template, b_template, h_treshold,
    def __init__(self, shot_area, icon_template, icon_treshold):
        """Constructor"""
        self.shot_area = shot_area
        self.icon_template = icon_template
        self.icon_treshold = icon_treshold
        self.screen = CutThePict(self.shot_area)
        self.icon_pos = {'top_left': (0,0), 'bottom_right': (0,0), 'center_point': (0,0)}

    def Find(self):
        icon_exist, self.icon_pos = FindObject(self.screen, self.icon_template, self.icon_treshold)
        #icon_exist = ColorExist(self.icon_area, self.icon_color_range)
        return icon_exist

    def Movement(self):
#chol_icon_area = {'top': wcen['y']-11, 'left': wcen['x']-552, 'width': 40, 'height': 40} #33:510 - 73:550
        i_top = self.icon_pos['top_left'][1]
        i_left = self.icon_pos['top_left'][0]
        i_width = self.icon_pos['bottom_right'][0]-self.icon_pos['top_left'][0]
        i_height = self.icon_pos['bottom_right'][1]-self.icon_pos['top_left'][1]
        icon_area = {'top': i_top, 'left': i_left, 'width': i_width, 'height': i_height}
        move = MoveCheck(icon_area)
        return move

    def Click(self):
        MoveCurAndClick(self.icon_pos['center_point'][0], self.icon_pos['center_point'][1])


class Vik_akk():
    login = ''
    def __init__(self):
        self.screenshot_area = {'top': 0, 'left': 0, 'width': scr_resolution['x'], 'height': scr_resolution['y']}

    def PromoOff(self):
        print('PromoOff start')
        header_substr = 'предложение'
        h_template = './templates/header_border_1.png'
        b_template = './templates/b_close.png'
        h_treshold = 0.5
        b_treshold = 0.99

        promo_window = Window(header_substr, self.screenshot_area, h_template, b_template, h_treshold, b_treshold)
        header_exist = promo_window.CheckHeader()
        if (header_exist):
            button_exist = promo_window.FindButton()
            if (button_exist):
                promo_window.PressButton()
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

            header_substr = 'лояльно'
            h_template = './templates/header_border_1.png'
            b_template = './templates/b_take.png'
            h_treshold = 0.5
            b_treshold = 0.99
            dl_window = Window(header_substr, self.screenshot_area, h_template, b_template, h_treshold, b_treshold)
            header_exist = dl_window.CheckHeader()
            if (header_exist):
                button_exist = dl_window.FindButton()
                if (button_exist):
                    dl_window.PressButton()
                else:
                    print('Кнопка не обнаружена')
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

            header_substr = 'участник'
            h_template = './templates/header_border_1.png'
            b_template = './templates/b_help.png'
            h_treshold = 0.5
            b_treshold = 0.99
            hlp_window = Window(header_substr, self.screenshot_area, h_template, b_template, h_treshold, b_treshold)
            header_exist = hlp_window.CheckHeader()
            if (header_exist):
                button_exist = hlp_window.FindButton()
                if (button_exist):
                    hlp_window.PressButton()
                else:
                    print('Кнопка не обнаружена')
                close_button_exist = hlp_window.FindButton()
        else:
            print('No icon')


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
                header_substr = 'оки'
                h_template = './templates/header_border_loki.png'
                b_template = './templates/b_take_2.png'
                h_treshold = 0.8
                b_treshold = 0.5
                chol_window = Window(header_substr, self.screenshot_area, h_template, b_template, h_treshold, b_treshold)
                header_exist = chol_window.CheckHeader()
                if (header_exist):
                    button_exist = chol_window.FindButton()
                    if (button_exist):
                        print('Кнопка найдена!')
                        chol_window.PressButton()
                    else:
                        print('Кнопка не найдена!')
            else:
                print('No movement')




Init()
print(wcen)
coolrock = Vik_akk()
coolrock.PromoOff()
coolrock.TakeDailyLoyalBonus()
coolrock.TakeChestOfLoki()
#coolrock.PressHelp()
#recognize_image()
#file_name = CutThePict({'top': 0, 'left': 0, 'width': scr_resolution['x'], 'height': scr_resolution['y']}, png=True)
#file_name = CutThePict({'top': 247, 'left': 830, 'width': 500, 'height': 42}, png=True)
#scr_s = CutThePict({'top': 0, 'left': 0, 'width': scr_resolution['x'], 'height': scr_resolution['y']})
#print(scr_s)
