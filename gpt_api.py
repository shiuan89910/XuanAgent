import logging
import time
import undetected_chromedriver as uc
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GPTAPI():
    def __init__(self, mail, pwd):
        """初始化 GPTAPI 類別，設定郵箱和密碼"""
        self.mail = mail
        self.pwd = pwd
        try:
            self.op = self._init_options()
            self.driver = self._init_driver()
        except Exception as e:
            logging.error(f"__init__() ERR: {e}")
            self.op = None
            self.driver = None

    def _init_options(self):
        """初始化 WebDriver 選項"""
        try:
            return webdriver.ChromeOptions()
        except Exception as e:
            logging.error(f"_init_options() ERR: {e}")
            return None

    def _init_driver(self):
        """初始化 WebDriver"""
        try:
            if self.op == None:
                return None
            self.op.add_argument("headless")
            self.op.add_argument(f"user-agent={UserAgent.random}")
            self.op.add_argument("user-data-dir=./")
            self.op.add_experimental_option("detach", True)
            self.op.add_experimental_option("excludeSwitches", ["enable-logging"])
            return uc.Chrome(chrome_options=self.op)
        except Exception as e:
            logging.error(f"_init_driver() ERR: {e}")
            return None
    
    def close(self):
        """關閉 WebDriver"""
        if self.driver:
            try:
                self.driver.close()
            finally:
                self.driver = None
                self.op = None

    def wait_find_elements(self, element, idx=0, sleep=0.01, timeout=10):
        """等待並找到元素"""
        try:
            WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((By.XPATH, element)))
            time.sleep(sleep)
            return self.driver.find_elements(By.XPATH, element)[idx]
        except:
            logging.error(f"The exception value is: {element}")
            return None

    def refresh(self, **kwargs):
        """刷新頁面"""
        idx = 0
        while True:
            try:
                new_chat = self.driver.find_elements(By.XPATH, kwargs["refresh"])[idx]
                new_chat.click() 
                time.sleep(kwargs.get("refresh_sec", 1))
                break
            except:
                idx += 1      

    def login(self, **kwargs):
        """登入操作"""
        try:
            self.driver.get(kwargs["url"])
            btn_login = self.wait_find_elements(kwargs["btn_login"])
            btn_login.click()
            btn_google = self.wait_find_elements(kwargs["btn_google"])
            btn_google.click()
            txt_mail = self.wait_find_elements(kwargs["txt_mail"])
            txt_mail.send_keys(self.mail)
            btn_mail = self.wait_find_elements(kwargs["btn_mail"])
            btn_mail.click()
            txt_pwd = self.wait_find_elements(kwargs["txt_pwd"], sleep=kwargs.get("btn_mail_sec", 7))
            txt_pwd.send_keys(self.pwd)
            btn_pwd = self.wait_find_elements(kwargs["btn_pwd"])
            btn_pwd.click()
            time.sleep(kwargs.get("btn_pwd_sec", 10))
            return True
        except Exception as e:
            logging.error(f"login() ERR: {e}")
            return False

    def get_output(self, input, timeout, **kwargs):
        """提交輸入並獲取輸出"""
        try:
            txt_input = self.wait_find_elements(kwargs["txt_input"])
            txt_input.send_keys(input)
            btn_input = self.wait_find_elements(kwargs["btn_input"])
            btn_input.click()
            text_output = self.wait_find_elements(kwargs["text_output"], idx=-1, timeout=timeout)
            return text_output.text
        except Exception as e:
            logging.error(f"get_output() ERR: {e}")
            return "Fail"


def main():
    url = input("請輸入 URL:")
    param_refresh = {
        "refresh": "//button[@class=\"text-token-text-primary\"]",
        "refresh_sec": 1,
        }
    
    param_login = {
        "url": url,
        "btn_login": "//button[@data-testid=\"login-button\"]",
        "btn_google": "//button[@data-provider=\"google\"]",
        "txt_mail": "//input[@type=\"email\"]",
        "btn_mail": "//button[@class=\"VfPpkd-LgbsSe VfPpkd-LgbsSe-OWXEXe-k8QpJ VfPpkd-LgbsSe-OWXEXe-dgl2Hf nCP5yc AjY5Oe DuMIQc LQeN7 qIypjc TrZEUc lw1w4b\"]",
        "btn_mail_sec": 7,
        "txt_pwd": "//input[@type=\"password\"]",
        "btn_pwd": "//button[@class=\"VfPpkd-LgbsSe VfPpkd-LgbsSe-OWXEXe-k8QpJ VfPpkd-LgbsSe-OWXEXe-dgl2Hf nCP5yc AjY5Oe DuMIQc LQeN7 qIypjc TrZEUc lw1w4b\"]",
        "btn_pwd_sec": 10,
        }
    
    param_output = {
        "txt_input": "//textarea[@id=\"prompt-textarea\"]",
        "btn_input": "//button[@data-testid=\"send-button\"]",
        "text_output": "//div[@class=\"markdown prose w-full break-words dark:prose-invert dark\"]",
        }
    
    # GPTAPI 流程
    ga = GPTAPI("mail", "password")
    if not ga.login(**param_login):
        logging.info("登入失敗，結束程式。")
        return
    while True:
        prompt = input("請輸入問題:").strip()
        if prompt.lower() == "close":
            logging.info("接收到關閉指令，結束循環。")
            break
        timeout_sec = input("請輸入回答的 Timeout 時間 (秒):").strip()
        timeout_sec = int(timeout_sec) if timeout_sec.isdigit() else 999
        result = ga.get_output(prompt, timeout_sec, **param_output)
        logging.info(result)
        ga.refresh(**param_refresh)
    ga.close()
    logging.info("程式結束，已關閉資源。")


if __name__ == "__main__":
    main()