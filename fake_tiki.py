



class TikiMini:

    MOTOR_MODE_PWM = 1
    MOTOR_MODE_PID = 2

    MOTOR_LEFT = -10
    MOTOR_RIGHT = 10

    def __init__():

        self.log_list = list()
        self.motor_mode = 0


    def set_led_color(num: int, r: int, g: int, b: int):

        print("set_led_color", num, r, g, b)

    
    def play_buzzer(freq: int):
        print("buzzer: ", freq)

    def stop_buzzer():
        print("buzzer stop")

    def log(log: str)
        self.log_list.append(log)
        self.log_list = self.log_list[:min(5, len(self.log_list))]
    
    def log_clear():
        self.log_list = list()

    def get_imu():
        return 0, 0, 0
    
    def get_current():
        return 0

    def get_battery_voltage():
        return 0

    def fire_cannon():
        print("BOOOOOOM!!!")
    
    def set_motor_mode(mode: int):
        self.motor_mode = mode

    def set_motor_power(motor: int, value: int):
        if self.motor_mode > 1:
            print("motor goes now", motor, value)