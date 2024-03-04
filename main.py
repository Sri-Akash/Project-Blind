import RPi.GPIO as GPIO
import time
from subprocess import Popen

GPIO.setmode(GPIO.BCM)

button1_pin = 17
button2_pin = 18

GPIO.setup(button1_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(button2_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while True:
    if GPIO.input(button1_pin) == GPIO.LOW:
        print("Button 1 pressed")
        Popen(['python', 'Image2Text.py'])

    if GPIO.input(button2_pin) == GPIO.LOW:
        print("Button 2 pressed")
        Popen(['python', 'distance_estimation.py'])

    time.sleep(0.1)
