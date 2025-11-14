from machine import I2C, Pin
import network
import socket
import json
import time

# ===== MPU6050 초기화 =====
i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=400000)
MPU_ADDR = 0x68

# 센서 초기화 (전원관리 레지스터)
try:
    i2c.writeto_mem(MPU_ADDR, 0x6B, bytes([0]))
except OSError:
    print("MPU6050 초기화 실패")

# ===== 센서 읽기 함수 =====
def safe_read(addr):
    try:
        data = i2c.readfrom_mem(MPU_ADDR, addr, 2)
        value = int.from_bytes(data, 'big')
        if value > 32768:
            value -= 65536
        return value
    except OSError:
        return 0  # 오류 발생 시 0 반환

def read_sensor():
    return {
        'AcX': safe_read(0x3B),
        'AcY': safe_read(0x3D),
        'AcZ': safe_read(0x3F),
        'GyX': safe_read(0x43),
        'GyY': safe_read(0x45),
        'GyZ': safe_read(0x47),
        'Temp': round(safe_read(0x41)/340 + 36.53, 2)
    }

# ===== SoftAP 설정 =====
def connect():
    wlan = network.WLAN(network.AP_IF)
    wlan.active(False)
    wlan.config(ssid='PicoW', password='12345678')
    wlan.active(True)
    return wlan.ifconfig()[0]

# ===== 소켓 서버 설정 =====
def open_socket():
    addr = ('0.0.0.0', 80)
    s = socket.socket()
    s.bind(addr)
    s.listen(2)
    s.settimeout(2)
    return s

# ===== 클라이언트 메시지 처리 =====
def handle_client(connection):
    try:
        client, addr = connection.accept()
        client.settimeout(2)
        try:
            _ = client.recv(1024)  # 요청 무시
        except OSError:
            pass
        sensor_data = read_sensor()
        mess = json.dumps(sensor_data)
        try:
            client.sendall(mess)
        except OSError:
            pass
        client.close()
    except OSError:
        pass

# ===== 메인 =====
try:
    ip = connect()
    print('AP IP:', ip)
    server = open_socket()
    print('Socket open')
    
    while True:
        handle_client(server)
        time.sleep(0.1)

except KeyboardInterrupt:
    server.close()
    print('Server closed')



