from machine import Pin,ADC
import urequests 
from machine import Pin 
import network
from time import sleep
c1 = 0
c2 = 0
total=0
led =Pin(1,Pin.OUT)
led2 =Pin(3,Pin.OUT)
ir1 = Pin(13, Pin.IN)
ir2 = Pin(18, Pin.IN)
temp=ADC(4)
 
HTTP_HEADERS = {'Content-Type': 'application/json'} 
THINGSPEAK_WRITE_API_KEY = 'IP0WXDALZ3K6CWYK'  
 
ssid = 'Redmi note 7 pro'
password = 'deba2211'
 
# Configure Pico W as Station
sta_if=network.WLAN(network.STA_IF)
sta_if.active(True)

if not sta_if.isconnected():
    print('connecting to network...')
    sta_if.connect(ssid, password)
    while not sta_if.isconnected():
     pass
print('network config:', sta_if.ifconfig()) 
 
while True:
    data=temp.read_u16()
    data1= data*0.000050354
    temp1=27-(data1-0.706)/0.001721
    if ir1.value()==0:
        c1 += 1
        sleep(0.5)
        print("in", c1)
    else:
        c1 = c1
        sleep(0.5)
        print("in",c1)
    if ir2.value()==0:
        c2 += 1
        sleep(0.5)
        print("out",c2) 
else:
        c2 = c2
        sleep(0.5)
        print("out",c2)
    total=c1-c2
    print("TOTAL", total)
    if total>0:
        led.value(1)
    else:
        led.value(0)
        
    if total>=10:
        led2.value(1)
    else:
        led2.value(0)
    
    
    dht_readings = {'field1':c1, 'field2':c2,'field3':total, 'field4':temp1} 
 request = urequests.post( 'http://api.thingspeak.com/update?api_key=' +  THINGSPEAK_WRITE_API_KEY, json = dht_readings, headers = HTTP_HEADERS )  

 request.close() 
 print(dht_readings) 
