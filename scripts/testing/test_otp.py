import json
from datetime import datetime

from Cryptodome.Cipher import AES
from Cryptodome.Hash import HMAC, SHA256
from Cryptodome.Util.Padding import unpad

IDEXPERT_REGISTER_KEY = bytes.fromhex('')


def decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    data = cipher.decrypt(data)
    data = unpad(data, AES.block_size)
    return data


def compute_otp(mac: bytes) -> str:
    i = mac[-1] & 15
    a = mac[i + 3] & 255
    b = (mac[i] & 127) << 24
    c = (mac[i + 1] & 255) << 16
    d = (mac[i + 2] & 255) << 8
    e = b | c
    f = e | d
    g = a | f
    return f'{g % 1000000:06d}'


def get_mobile_otp(push_key: str) -> str:
    push_key = bytes.fromhex(push_key)
    timestamp = int(datetime.now().timestamp() / 30)
    timestamp = f'{timestamp:016X}'
    timestamp = bytes.fromhex(timestamp)
    mac = HMAC.new(push_key, timestamp, SHA256).digest()
    otp = compute_otp(mac)
    return otp


def get_seconds_until_next_otp() -> int:
    return 30 - int(datetime.now().timestamp() % 30)


def parse_initial_key(initial_key: str) -> dict[str, str | int]:
    iv = bytes.fromhex(initial_key[:32])
    data = bytes.fromhex(initial_key[32:])
    data = decrypt(IDEXPERT_REGISTER_KEY, iv, data)
    data = json.loads(data)
    return data


def compute_push_key(ik: str, device_id: str) -> str:
    hex1 = SHA256.new(ik.encode()).hexdigest()
    hex2 = SHA256.new(device_id.encode()).hexdigest()
    return hex1[:32] + hex2[:32]


if __name__ == '__main__':
    # Example usage
    inital_key = ""
    device_id = ""

    data = parse_initial_key(inital_key)
    ik = data['ik']
    push_key = compute_push_key(ik, device_id)
    otp = get_mobile_otp(push_key)
    print(otp)