import pygame
import time

pygame.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
js.init()

print('按 RT/LT 扳机测试，Ctrl+C 退出')
print('RT 通常是 axis 5，LT 通常是 axis 2')

while True:
    pygame.event.pump()
    for i in range(js.get_numaxes()):
        val = js.get_axis(i)
        if abs(val) > 0.1:
            print(f'Axis {i}: {val:.2f}')
    time.sleep(0.1)
