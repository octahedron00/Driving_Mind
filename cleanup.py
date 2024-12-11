
from tiki.mini import TikiMini


pub = TikiMini()


pub.set_motor_power(pub.MOTOR_LEFT, 0)
pub.set_motor_power(pub.MOTOR_RIGHT, 0)
pub.stop_buzzer()
for i in range(8):
    pub.set_led_color(i, 50, 50, 50)