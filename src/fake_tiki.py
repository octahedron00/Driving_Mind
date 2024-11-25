'''
    가짜 Tiki를 만들어, 함수가 전체적으로 잘 작동하는지 확인.
    오탈자 없는지 자동으로 확인해줌.

    대신 실제 robot 쓸 때는 이 패키지들 싹 바꿔줘야 합니다

'''

class TikiMini:

    MOTOR_MODE_PWM = 1
    MOTOR_MODE_PID = 2

    MOTOR_LEFT = -10
    MOTOR_RIGHT = 10

    def __init__(self):

        self.log_list = list()
        self.motor_mode = 0


    def set_led_color(self, num: int, r: int, g: int, b: int):

        print("set_led_color", num, r, g, b)

    
    def play_buzzer(self, freq: int):
        print("buzzer: ", freq)

    def stop_buzzer(self):
        print("buzzer stop")

    def log(self, log: str):
        # log: max 5 lines 

        self.log_list.append(log)
        self.log_list = self.log_list[max(0, len(self.log_list)-5):]
        
        print("^--------- LOG ---------^")
        for log in self.log_list:
            print(log)
        print("|--------- LOG ---------|")

    
    def log_clear(self):
        self.log_list = list()

    def get_imu(self):
        return 0, 0, 0
    
    def get_current(self):
        return 300

    def get_battery_voltage(self):
        return 11

    def fire_cannon(self):
        print("BOOOOOOM!!!")
    
    def set_motor_mode(self, mode: int):
        self.motor_mode = mode

    def set_motor_power(self, motor: int, value: int):
        if self.motor_mode > 1:
            # print("motor goes now", motor, value)
            pass
        else:
            raise Exception
    
    def get_encoder(self):
        return 0, 0
    