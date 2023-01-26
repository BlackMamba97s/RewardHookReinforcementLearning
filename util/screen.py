from PIL import ImageGrab



screen_width = 800
screen_height = 600
leftx = 0
lefty = 20
screen_pos = [[leftx, lefty], [leftx + screen_width,
                               lefty + screen_height]]



def get_screen(screen_pos):

    left_x, left_y = screen_pos[0]
    right_x, right_y = screen_pos[1]
    image = ImageGrab.grab((left_x, left_y, right_x, right_y))
    return image