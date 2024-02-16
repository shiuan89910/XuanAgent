import glob
import logging
import numpy as np
import os
#import pytesseract
import re
import time
import torch
import undetected_chromedriver as uc
from fake_useragent import UserAgent
from modelscope.pipelines import pipeline as ms_pipeline
from modelscope.utils.constant import Tasks
from PIL import Image, ImageEnhance, ImageFilter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


proj_name = "XuanAgent"
try:
    user_name = os.getlogin()
    root_path = f"C:\\Users\\{user_name}\\{proj_name}"
except:
    root_path = f"/{proj_name}"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 透過 otsu 算法計算圖片的二值化閾值
def otsu_thresh(image):
    try:
        pixel_counts = np.bincount(image.ravel(), minlength=256)
        total = image.size
        sum_total = np.dot(np.arange(256), pixel_counts)
        sumB = 0.0
        wB = 0.0
        max_variance = 0.0
        threshold = 0
        for i in range(256):
            wB += pixel_counts[i]
            if wB == 0 or wB == total:
                continue
            wF = total - wB
            sumB += i * pixel_counts[i]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            variance_between = wB * wF * (mB - mF) ** 2
            if variance_between > max_variance:
                max_variance = variance_between
                threshold = i
        return threshold
    except Exception as e:
        logging.error(f"otsu_thresh() ERR: {e}")
        return 0


# 驗證碼預處理
def pre_proc_captcha(**kwargs):
    try:
        image = Image.open(kwargs["img_captcha_path"]).crop(kwargs["img_captcha_crop_loc_2"]) if kwargs["img_captcha_recapture"] else Image.open(kwargs["img_captcha_path"]).crop(kwargs["img_captcha_crop_loc"])
        if not kwargs["use_rgb"]:
            image = image.convert(kwargs["img_captcha_pre_proc_grey"])
            image = Image.fromarray(np.invert(np.array(image) > otsu_thresh(np.array(image))).astype(np.uint8) * 255)
            image = ImageEnhance.Contrast(image).enhance(kwargs["img_captcha_pre_proc_enhance"])
            image = image.filter(ImageFilter.SHARPEN)
            image = image.filter(ImageFilter.MedianFilter(size=kwargs["img_captcha_pre_proc_filter"])) 
        return image
    except Exception as e:
        logging.error(f"pre_proc_captcha() ERR: {e}")
        return None


"""
# 進行驗證碼 OCR
def ocr_captcha(image, **kwargs):
    try:
        text = pytesseract.image_to_string(image, lang=kwargs["img_captcha_pre_proc_txt_lang"], config=kwargs["img_captcha_pre_proc_txt_config"])
        #image.save(kwargs["img_captcha_path"])
        #logging.info("\n" + text + "\n") 
        return re.sub(r"[^a-zA-Z0-9]", "", text.strip())
    except Exception as e:
        logging.error(f"ocr_captcha() ERR: {e}")
        return ""
"""


# 預訓練模型進行驗證碼 OCR
def ocr_captcha_2_from_pretrained(**kwargs):
    try:
        pipe = ms_pipeline(Tasks.ocr_recognition, model=kwargs["model"])
        return pipe
    except Exception as e:
        logging.error(f"ocr_captcha_2_from_pretrained() ERR: {e}")
        return None

def ocr_captcha_2(image, pipe):
    try:
        result = pipe(image)
        #logging.info(result) 
        return re.sub(r"[^a-zA-Z0-9]", "", result["text"][0].upper().strip())
    except Exception as e:
        logging.error(f"ocr_captcha_2() ERR: {e}")
        return ""


# 預訓練模型進行驗證碼 STT 語音到文字
def stt_from_pretrained(**kwargs):
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            kwargs["model_id"],
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=kwargs["low_cpu_mem_usage"],
            use_safetensors=kwargs["use_safetensors"],
            use_flash_attention_2=kwargs["use_flash_attention_2"], # Flash Attention
            )
        model = model.to_bettertransformer() if kwargs["use_SPDA"] else model # Torch Scale-Product-Attention (SDPA)
        model.to(device)
        processor = AutoProcessor.from_pretrained(kwargs["model_id"])
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=kwargs["max_new_tokens"],
            chunk_length_s=kwargs["chunk_length_s"],
            batch_size=kwargs["batch_size"],
            return_timestamps=kwargs["return_timestamps"],
            torch_dtype=torch_dtype,
            device=device,
            )  
        return pipe
    except Exception as e:
        logging.error(f"stt_from_pretrained() ERR: {e}")
        return None
    
# 使用 STT 生成結果
def stt_generate(data_src, pipe, **kwargs):
    try:
        result = pipe(
            data_src,
            return_timestamps=kwargs["return_timestamps"],
            generate_kwargs={"language": kwargs["language"], "task": kwargs["task"]},
            )
        return result[kwargs["ret_format"]]
    except Exception as e:
        logging.error(f"stt_generate()  ERR: {e}")
        return None


class BookTicket():
    def __init__(self, id, phone, ocr_pipe=None, stt_pipe=None):
        """初始化預訂票務類，設定用戶資訊與 OCR、STT 管道"""
        self.id = id
        self.phone = phone
        self.ocr_pipe = ocr_pipe
        self.stt_pipe = stt_pipe
        self.op = None
        self.driver = None
        self._setup()

    def _setup(self):
        """初始化設定，包括 webdriver 選項與驅動"""
        self.op = self._init_options()
        if self.op:
            self.driver = self._init_driver()
        else:
            self.ocr_pipe = None
            self.stt_pipe = None

    def _init_options(self):
        """初始化 webdriver 選項"""
        try:
            return webdriver.ChromeOptions()
        except Exception as e:
            logging.error(f"_init_options() ERR: {e}")
            return None

    def _init_driver(self):
        """初始化 webdriver """
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
        """關閉 webdriver"""
        if self.driver:
            try:
                self.driver.close()
            finally:
                self.stt_pipe = None
                self.ocr_pipe = None
                self.driver = None
                self.op = None

    def get_code(self):
        """獲取車次代碼"""
        return self.code

    def time_to_minutes(self, time_str):
        """將時間字串轉換為分鐘"""
        hours, minutes = map(int, time_str.split(":"))
        return hours * 60 + minutes

    def wait_find_elements(self, element, idx=0, sleep=0.01, timeout=10):
        """等待並找到指定元素"""
        try:            
            WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((By.XPATH, element)))
            time.sleep(sleep)
            elements = self.driver.find_elements(By.XPATH, element)
            return elements if idx is None else elements[idx]
        except:
            logging.error(f"The exception value is: {element}")
            return None

    def book_pg_0(self, **kwargs):
        """訪問首頁並同意隱私政策"""
        try:
            self.driver.get(kwargs["url"])
            btn_agree = self.wait_find_elements(kwargs["btn_agree"], sleep=1)
            btn_agree.click()
            return True
        except Exception as e:
            logging.error(f"book_pg_0() ERR: {e}")
            return False

    def book_pg_1(self, **kwargs):
        """選擇出發和抵達站點，日期和人數，並處理驗證碼"""
        try:
            select_departure = self.wait_find_elements(kwargs["select_departure"])
            Select(select_departure).select_by_value(kwargs["departure"])
            select_arrival = self.wait_find_elements(kwargs["select_arrival"])
            Select(select_arrival).select_by_value(kwargs["arrival"])
            select_date = self.wait_find_elements(kwargs["select_date"])
            select_date.click()
            confirm_date = self.wait_find_elements(kwargs["confirm_date"])
            confirm_date.click()
            select_people = self.wait_find_elements(kwargs["select_people"])
            Select(select_people).select_by_value(kwargs["people"])
            if kwargs["use_train_code"]:
                radio_train_code = self.wait_find_elements(kwargs["radio_train_code"])
                radio_train_code.click()
                if not kwargs["img_captcha_recapture"]:
                    txt_train_code = self.wait_find_elements(kwargs["txt_train_code"])
                    txt_train_code.send_keys(kwargs["train_code"])
            else:
                select_time = self.wait_find_elements(kwargs["select_time"])
                Select(select_time).select_by_value(kwargs["time"])   
            if kwargs["captcha_method"] == 0:
                for wav_file in glob.glob(os.path.join(kwargs["speech_captcha_path"], "*.wav")):
                    os.remove(wav_file)
                speech_captcha = self.wait_find_elements(kwargs["speech_captcha"])
                speech_captcha.click()
                time.sleep(kwargs["speech_captcha_sec"])
                wav_files = sorted(glob.glob(os.path.join(kwargs["speech_captcha_path"], "*.wav")), key=lambda x: os.path.getmtime(x), reverse=True)
                if wav_files:
                    ret = stt_generate(wav_files[0], self.stt_pipe, **kwargs)
                else:
                    return False
                from_method_captcha = re.sub(r"[^a-zA-Z0-9]", "", ret[0]["text"].strip())
                #logging.info(from_method_captcha)
            elif kwargs["captcha_method"] == 1:
                self.driver.save_screenshot(kwargs["img_captcha_path"])
                #img_captcha = pre_proc_captcha(**kwargs)
                #from_method_captcha = ocr_captcha(img_captcha, **kwargs)
                img_captcha = pre_proc_captcha(**kwargs)
                from_method_captcha = ocr_captcha_2(img_captcha, self.ocr_pipe)
            else:
                from_method_captcha = input("請輸入圖中的字:")
            txt_captcha = self.wait_find_elements(kwargs["txt_captcha"])
            txt_captcha.send_keys(from_method_captcha)
            btn_submit = self.wait_find_elements(kwargs["btn_submit"])
            btn_submit.click()
            return True
        except Exception as e:
            logging.error(f"book_pg_1() ERR: {e}")
            return False  
        
    def book_pg_2(self, **kwargs):
        """提交訂票請求並檢查結果"""
        try:
            time.sleep(kwargs["sleep_sec"])
            submit_result = self.wait_find_elements(kwargs["submit_result"])
            if submit_result.text == kwargs["submit_result_Fail"]:
                return False
            else:
                return True
        except Exception as e:
            logging.error(f"book_pg_2() ERR: {e}")
            return True

    def book_pg_3(self, **kwargs):
        """選擇列車和時間"""
        try:
            time.sleep(kwargs["sleep_sec"])
            train_group = self.wait_find_elements(kwargs["train_group"], idx=None)
            time_within_range = [t for t in train_group if self.time_to_minutes(kwargs["time_start"]) <= self.time_to_minutes(t.get_attribute(kwargs["train_group_departure"])) <= self.time_to_minutes(kwargs["time_end"])]
            if time_within_range:
                minutes_list = [self.time_to_minutes(t.get_attribute(kwargs["train_group_estimated"])) for t in time_within_range]
                list_idx = max(minutes_list) if kwargs.get("max_time", False) else min(minutes_list)
                target_element = time_within_range[minutes_list.index(list_idx)]
                target_element.click()
                self.code = target_element.get_attribute(kwargs["train_group_code"]) 
            else:
                return False
            btn_submit = self.wait_find_elements(kwargs["btn_submit"])
            btn_submit.click()
            return True
        except Exception as e:
            logging.error(f"book_pg_3() ERR: {e}")
            return False
        
    def book_pg_4(self, **kwargs):
        """填寫乘客資訊並提交"""
        try:
            time.sleep(kwargs["sleep_sec"])
            txt_id = self.wait_find_elements(kwargs["txt_id"])
            txt_id.send_keys(self.id)
            txt_phone = self.wait_find_elements(kwargs["txt_phone"])
            txt_phone.send_keys(self.phone)
            radio_agree = self.wait_find_elements(kwargs["radio_agree"])
            radio_agree.click()
            btn_submit = self.wait_find_elements(kwargs["btn_submit"])
            btn_submit.click()
            return True
        except Exception as e:
            logging.error(f"book_pg_4() ERR: {e}")
            return False
        
    def book_pg_5(self, **kwargs):
        """檢查訂票結果並保存截圖"""
        try:
            time.sleep(kwargs["sleep_sec"])
            details = self.wait_find_elements(kwargs["details"])
            details_info = details.find_element(By.TAG_NAME, kwargs["details_tag_name"])
            if kwargs["details_success"] in details_info.text:
                self.driver.save_screenshot(kwargs["img_details_path"])
                logging.info("訂位成功")
                return True
            else:
                logging.info("訂位失敗")
                return False
        except Exception as e:
            logging.error(f"book_pg_5() ERR: {e}")
            return False


def main(**kwargs):
    url = input("請輸入 URL:")
    param_book_pg_0 = {
        # 預訂票務第 0 頁參數
        "url": url,
        "btn_agree": "//button[@class=\"policy-btn-accept\"]",
        }

    param_book_pg_1 = {
        # 預訂票務第 1 頁參數
        "select_departure": "//select[@name=\"selectStartStation\"]",
        "departure": kwargs["departure"],
        "select_arrival": "//select[@name=\"selectDestinationStation\"]",
        "arrival": kwargs["arrival"],
        "select_date": "//input[@class=\"uk-input\"]",
        "confirm_date": "//span[@aria-label=\"" + kwargs["confirm_date"] + "\"]",
        "select_people": "//select[@name=\"ticketPanel:rows:0:ticketAmount\"]",
        "people": kwargs["people"],
        "use_train_code": kwargs["use_train_code"],
        "radio_train_code": "//input[@data-target=\"search-by-trainNo\"]",
        "txt_train_code": "//input[@name=\"toTrainIDInputField\"]",
        "train_code": kwargs["train_code"],
        "select_time": "//select[@name=\"toTimeTable\"]",
        "time": kwargs["time"],
        "captcha_method": kwargs["captcha_method"],
        "img_captcha_path": kwargs["img_captcha_path"],
        "use_rgb": True,
        "img_captcha_recapture": False,
        "img_captcha_crop_loc": (1249, 519, 1391, 576),
        "img_captcha_crop_loc_2": (1235, 611, 1391, 677),
        "img_captcha_pre_proc_grey": "L",
        "img_captcha_pre_proc_enhance": 2,
        "img_captcha_pre_proc_filter": 3,
        "img_captcha_pre_proc_txt_lang": "eng",
        "img_captcha_pre_proc_txt_config": "--psm 6 --oem 3",
        "speech_captcha": "//span[@title=\"語音播放\"]",
        "speech_captcha_sec": kwargs["speech_captcha_delay_sec"],
        "speech_captcha_path": kwargs["speech_captcha_path"],
        "return_timestamps": True, # return_timestamps=True or return_timestamps="word"
        "language": "english", # "language": "english" or None
        "task": "translate", # "task": "translate" or None
        "ret_format": "chunks", # "chunks" or "text"
        "txt_captcha": "//input[@name=\"homeCaptcha:securityCode\"]",
        "btn_submit": "//input[@name=\"SubmitButton\"]",
        }

    param_book_pg_2 = {
        # 預訂票務第 2 頁參數
        "sleep_sec": kwargs["pg_2_delay_sec"],
        "submit_result": "//span[@class=\"material-icons icon-alert\"]",
        "submit_result_Fail": "error",
        }

    param_book_pg_3 = {
        # 預訂票務第 3 頁參數
        "sleep_sec": kwargs["pg_3_delay_sec"],
        "train_group": "//input[@name=\"TrainQueryDataViewPanel:TrainGroup\"]",
        "time_start": kwargs["time_start"],
        "time_end": kwargs["time_end"],
        "train_group_departure": "querydeparture",
        "train_group_estimated": "queryestimatedtime",
        "train_group_code": "querycode",
        "max_time": kwargs["max_time"],
        "btn_submit": "//input[@name=\"SubmitButton\"]",
        }

    param_book_pg_4 = {
        # 預訂票務第 4 頁參數
        "sleep_sec": kwargs["pg_4_delay_sec"],
        "txt_id": "//input[@id=\"idNumber\"]",
        "txt_phone": "//input[@id=\"mobilePhone\"]",
        "radio_agree": "//input[@name=\"agree\"]",
        "btn_submit": "//input[@id=\"isSubmit\"]",
        }

    param_book_pg_5 = {
        # 預訂票務第 5 頁參數
        "sleep_sec": kwargs["pg_5_delay_sec"],
        "details": "//p[@class=\"alert-title\"]",
        "details_tag_name": "span",
        "details_success": "您已完成訂位",
        "img_details_path": kwargs["img_details_path"],
        }

    param_sp = {
        # 語音識別參數
        "model_id": f"{root_path}\\whisper-medium",
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "use_flash_attention_2": False,
        "use_SPDA": False,
        "max_new_tokens": kwargs["stt_max_new_tokens"],
        "chunk_length_s": kwargs["stt_chunk_length_s"],
        "batch_size": kwargs["stt_batch_size"],
        "return_timestamps": True, # return_timestamps=True or return_timestamps="word"
        }

    param_op = {
        # OCR 識別參數
        "model": f"{root_path}\\ocr-captcha\output_small",
        }

    #pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if param_book_pg_1["captcha_method"] == 0:
        ocr_pipe = None
        stt_pipe = stt_from_pretrained(**param_sp)
    elif param_book_pg_1["captcha_method"] == 1:
        ocr_pipe = ocr_captcha_2_from_pretrained(**param_op)
        stt_pipe = None
    else:
        ocr_pipe = None
        stt_pipe = None

    # 預訂票務流程
    while True:
        bt = BookTicket(
            id=kwargs["id"],
            phone=kwargs["phone"],
            ocr_pipe=ocr_pipe, 
            stt_pipe=stt_pipe,
            )
        if not bt.book_pg_0(**param_book_pg_0):
            bt.close()
            continue
        if not bt.book_pg_1(**param_book_pg_1):
            bt.close()
            continue
        ctn = False
        while not bt.book_pg_2(**param_book_pg_2):
            param_book_pg_1["img_captcha_recapture"] = True
            if not bt.book_pg_1(**param_book_pg_1):
                ctn = True
                bt.close()
                break
        if ctn:
            continue
        if not param_book_pg_1["use_train_code"] and not bt.book_pg_3(**param_book_pg_3):
            bt.close()
            continue
        if not bt.book_pg_4(**param_book_pg_4):
            bt.close()
            continue
        succ = bt.book_pg_5(**param_book_pg_5)
        code = bt.get_code()
        bt.close()
        if succ:
            break
    return code


config = {
    # 預訂票務配置設定
    "id": "id", ##### 身分證 #####
    "phone": "phone", ##### 手機 #####
    "departure": "2", ##### 站別 (出發) #####
    "arrival": "7", ##### 站別 (抵達) #####
    "confirm_date": "一月 1, 2024", ##### 日期 #####
    "people": "1F", ##### 人數 #####

    "use_train_code": False, ##### 依 [時間 (False) / 車次 (True)] #####
    "train_code": "669", ##### 車次 #####
    "time": "600P", ##### 時間 #####
    "time_start": "18:00", ##### 時間範圍 (開始) #####
    "time_end": "20:00", ##### 時間範圍 (結束) #####
    "max_time": False, ##### 選時程長短 [最短 (False) / 最長 (True)] #####

    # 預訂票務配置設定 (進階)
    "captcha_method": 0, ##### 驗證方式 #####
    "img_captcha_path": f"{root_path}\\captcha.png",
    "use_rgb": True,
    "speech_captcha_path": f"C:\\Users\\{user_name}\\Downloads",
    "img_details_path": f"{root_path}\\details.png",
    "speech_captcha_delay_sec": 1,
    "pg_2_delay_sec": 1,
    "pg_3_delay_sec": 1,
    "pg_4_delay_sec": 1,
    "pg_5_delay_sec": 3,
    "stt_max_new_tokens": 8,
    "stt_chunk_length_s": 8,
    "stt_batch_size": 8,
    }


if __name__ == "__main__":
    main(**config)