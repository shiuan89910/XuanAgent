import logging
from serpapi import GoogleSearch


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GoogleSearchAPI():
    def __init__(self, api_key):
        """初始化 GoogleSearchAPI 類別，設定 API 金鑰"""
        self.api_key = api_key

    def search(self, query, **kwargs):
        """執行 Google 搜索請求，返回搜索結果"""
        try:
            param_search = {
                "api_key": self.api_key,
                "q": query,
                "type": kwargs.get("type", "web"), # "web" or "image" or "news"
                "device": kwargs.get("device", "desktop"), # "desktop" or "mobile"
                "location": kwargs.get("location"), # "taipei" or "taichung" or "tainan" or "kaohsiung"
                "google_domain": kwargs.get("google_domain", "google.com.tw"), # "google.com" or "google.com.tw"
                "gl": kwargs.get("gl", "tw"), # "us" or "tw"
                "hl": kwargs.get("hl", "tw"), # "en" or "tw"
                "page": kwargs.get("page", 1), # 1 ~ 10
                "num": kwargs.get("num", 10), # 1 ~ 100
                "uule": kwargs.get("uule"),
                "output": kwargs.get("output", "json"), # "json" or "html"   
                }
            search = GoogleSearch(param_search)
            return search.get_dict()
        except Exception as e:
            logging.error(f"search() ERR: {e}")
            return None


# 創建 GoogleSearchAPI 實例並執行搜索
def main():
    param_search = {
        "type": "web",
        "device": "desktop",
        "location": None,
        "google_domain": "google.com.tw",
        "gl": "tw",
        "hl": "tw",
        "page": 1,
        "num": 10,
        "uule": None,
        "output": "json",
        }
    gsa = GoogleSearchAPI("api_key")
    ret = gsa.search("huggingface", **param_search)
    logging.info(ret)


if __name__ == "__main__":
    main()