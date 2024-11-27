# 실제 메일 전송 기능
# segmentation

from twilio.rest import Client

def send_sms(to_number, body):
    # Twilio 계정 정보
    account_sid = "your_account_sid"
    auth_token = "your_auth_token"
    client = Client(account_sid, auth_token)

    try:
        # 메시지 보내기
        message = client.messages.create(
            body=body,
            from_="your_twilio_phone_number",  # Twilio에서 제공한 전화번호
            to=to_number
        )
        print(f"메시지 전송 완료: SID {message.sid}")
    except Exception as e:
        print(f"메시지 전송 실패: {e}")

# 예시 호출
send_sms("+821012345678", "위험 상황이 감지되었습니다. 즉시 확인하세요.")

def read_token_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # 파일에서 토큰 읽기
            token = file.readline().strip()
            return token
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return None

# 예시 사용법
file_path = '/path/to/your/file.txt'  # 파일의 절대 경로나 상대 경로를 입력하세요.
token = read_token_from_file(file_path)
if token:
    print(f"토큰 값: {token}")
else:
    print("토큰 값을 읽어오지 못했습니다.")

