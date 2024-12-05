from twilio.rest import Client

class Alert:
    def __init__(self, account_sid_path, auth_token_path):
        self.account_sid = self.get_credential_from(account_sid_path)
        self.auth_token = self.get_credential_from(auth_token_path)
        self.from_number = "+17753699968"  # Twilio에서 제공한 인증된 발신 번호

        if self.account_sid and self.auth_token:
            print("Credential 값을 읽어왔습니다.")
        else:
            print("Credential 값을 읽어오지 못했습니다.")

        self.client = Client(self.account_sid, self.auth_token)

    def send_sms(self, to_number, body):
        if not self.account_sid or not self.auth_token:
            print("Twilio credentials가 없습니다. 메시지를 보낼 수 없습니다.")
            return

        try:
            # 메시지 보내기
            message = self.client.messages.create(
                body=f"[고령자 위험 감지 시스템] {body}",  # 시스템 이름을 본문에 포함
                from_=self.from_number,  # Twilio 인증 발신 번호
                to=to_number
            )
            print(f"메시지 전송 완료: SID {message.sid}")
        except Exception as e:
            print(f"메시지 전송 실패: {e}")

    @staticmethod
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
