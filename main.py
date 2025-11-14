from machine import I2C, Pin
import network
import socket
import json
import time

# ==========================================================
#  MPU6050 x 2 + Raspberry Pi Pico W SoftAP 통합 예제
#  - Pico W가 SoftAP(와이파이 공유기) 역할 수행
#  - 클라이언트가 접속 요청을 보내면 두 개의 MPU6050 센서 데이터 반환
# ==========================================================

# ===============================
# ① I2C 설정 및 센서 스캔
# ===============================
i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=400000)   # I2C0, GP1=SCL, GP0=SDA, 400kHz

print("Scanning I2C bus...")
scan = i2c.scan()
print("Scan result:", [hex(a) for a in scan])

MPU_ADDR1 = 0x68   # AD0 = GND
MPU_ADDR2 = 0x69   # AD0 = 3.3V

if MPU_ADDR1 in scan:
    print("Found MPU #1 (0x68)")
else:
    print("MPU #1 (0x68) not detected")

if MPU_ADDR2 in scan:
    print("Found MPU #2 (0x69)")
else:
    print("MPU #2 (0x69) not detected")

# 실제로 초기화에 성공한 센서 주소들을 저장
mpu_addrs = []

for addr in (MPU_ADDR1, MPU_ADDR2):
    if addr in scan:
        try:
            # PWR_MGMT_1(0x6B) = 0 → sleep 해제
            i2c.writeto_mem(addr, 0x6B, bytes([0]))
            mpu_addrs.append(addr)
            print("MPU init OK at", hex(addr))
        except OSError:
            print("MPU init FAILED at", hex(addr))

print("Active MPU addrs:", [hex(a) for a in mpu_addrs])

# ===============================
# ② 센서 데이터 읽기 함수 (공통)
# ===============================
def safe_read(dev_addr, reg_addr):
    """
    지정한 MPU6050(dev_addr)의 레지스터(reg_addr)에서 2바이트를 안전하게 읽는다.
    - 통신 오류 발생 시 0 반환
    """
    try:
        data = i2c.readfrom_mem(dev_addr, reg_addr, 2)
        value = int.from_bytes(data, 'big')
        if value > 32768:   # 16비트 signed 변환
            value -= 65536
        return value
    except OSError:
        return 0

def read_one_mpu(dev_addr):
    """
    하나의 MPU6050(dev_addr)에 대해 가속도, 자이로, 온도 값을 dict로 반환
    """
    return {
        'AcX': safe_read(dev_addr, 0x3B),
        'AcY': safe_read(dev_addr, 0x3D),
        'AcZ': safe_read(dev_addr, 0x3F),
        'GyX': safe_read(dev_addr, 0x43),
        'GyY': safe_read(dev_addr, 0x45),
        'GyZ': safe_read(dev_addr, 0x47),
        'Temp': round(safe_read(dev_addr, 0x41) / 340 + 36.53, 2),
        'addr': hex(dev_addr),
    }

def read_all_sensors():
    """
    활성화된 모든 MPU6050(mpu_addrs 기준)을 읽어서
    mpu1, mpu2 형태의 dict로 반환
    """
    result = {}
    for idx, addr in enumerate(mpu_addrs, start=1):
        key = f"mpu{idx}"   # mpu1, mpu2 ...
        result[key] = read_one_mpu(addr)
    return result

# ===============================
# ③ SoftAP(Access Point) 구성
# ===============================
def connect():
    """
    Pico W를 WiFi AP(핫스팟) 모드로 활성화
    - SSID: PicoW
    - PASSWORD: 12345678
    """
    wlan = network.WLAN(network.AP_IF)
    wlan.active(False)
    wlan.config(ssid='PicoW', password='12345678')
    wlan.active(True)

    return wlan.ifconfig()[0]   # AP IP 주소 반환

# ===============================
# ④ 소켓 서버 생성
# ===============================
def open_socket():
    """
    포트 80에서 HTTP 요청을 수신하도록 소켓 오픈
    - SoftAP 상태의 PicoW가 간단한 서버 기능 수행
    """
    addr = ('0.0.0.0', 80)
    s = socket.socket()
    s.bind(addr)
    s.listen(2)
    s.settimeout(2)
    return s

# ===============================
# ⑤ 클라이언트 처리
# ===============================
def handle_client(connection):
    """
    클라이언트가 접속하면:
    - 요청 수신 (내용은 사용하지 않음)
    - 두 개의 MPU6050 센서 값을 JSON으로 인코딩하여 전송
    """
    try:
        client, addr = connection.accept()
        client.settimeout(2)

        # 요청 데이터 읽기 (내용은 무시)
        try:
            _ = client.recv(1024)
        except OSError:
            pass

        # 센서 데이터 전송 (mpu1, mpu2 포함)
        sensor_data = read_all_sensors()
        mess = json.dumps(sensor_data)

        try:
            client.sendall(mess)
        except OSError:
            pass

        client.close()
    except OSError:
        pass

# ===============================
# ⑥ 메인 루프
# ===============================
try:
    ip = connect()
    print('AP IP:', ip)

    server = open_socket()
    print('Socket open')

    # 계속해서 연결 요청 처리 + 센서 값 콘솔 출력
    while True:
        # 콘솔에 두 센서 데이터 출력
        sensor_data = read_all_sensors()
        print("Sensor:", sensor_data)

        # 클라이언트가 접속하면 JSON 응답
        handle_client(server)

        time.sleep(0.1)

except KeyboardInterrupt:
    server.close()
    print('Server closed')