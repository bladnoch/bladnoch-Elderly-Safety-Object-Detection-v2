# 실제 메일 전송 기능
# segmentation

from twilio.rest import Client

def send_sms(to_number, body):
    # Twilio 계정 정보
    account_sid = get_credential_from('/Users/doungukkim/Desktop/workspace/object-detecting-v2/credentials/account_sid.txt')
    auth_token=get_credential_from('/Users/doungukkim/Desktop/workspace/object-detecting-v2/credentials/auth_token.txt')

    if auth_token and account_sid:
        print("credential 값을 읽어왔습니다.")
    else:
        print("credential 값을 읽어오지 못했습니다.")

    client = Client(account_sid, auth_token)

    try:
        # 메시지 보내기
        message = client.messages.create(
            body=body,
            from_="given_phone_number",  # Twilio에서 제공한 전화번호
            to=to_number
        )
        print(f"메시지 전송 완료: SID {message.sid}")
    except Exception as e:
        print(f"메시지 전송 실패: {e}")


def get_credential_from(file_path):
    try:
        with open(file_path, 'r') as file:
            token = file.readline().strip()
            return token
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return None

# 예시 호출
send_sms("+01046054434", "위험 상황이 감지되었습니다. 즉시 확인하세요.")
