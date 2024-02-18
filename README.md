# 0. 目錄



# 1. XuanAgent 專案簡介
透過大型語言模型智能代理，用於加強 AI 系統的決策能力，提升代理的學習效率和適應性。
- 大型語言模型（Large Language Model, LLM）驅動的自主代理系統
本文概述了 LLM 驅動的自主代理系統的關鍵組件和功能。更多詳細信息，請參考[Lilian Weng's Blog](https://lilianweng.github.io/posts/2023-06-23-agent/)。

- 在一個 LLM 驅動的自主代理（Autonomous Agent）系統中，LLM 充當代理的大腦，並由幾個關鍵組件：
![Overview of a LLM-powered autonomous agent system](https://lilianweng.github.io/posts/2023-06-23-agent/agent-overview.png "Overview of a LLM-powered autonomous agent system")


## 1.1. 規劃（Planning）
- **子目標和分解（Subgoal and decomposition）**：代理將大任務分解成較小、可管理的子目標，使複雜任務的處理更高效。
- 
- **反思和改進（Reflection and refinement）**：代理能夠自我批評和反思過去的行動，從錯誤中學習並為未來的步驟進行改進，從而提高最終結果的質量。


## 1.2. 記憶（Memory）
- **短期記憶（Short-term memory）**：視作模型利用短期記憶進行上下文學習。
- 
- **長期記憶（Long-term memory）**：使代理能夠在長時間內保留並回憶（無限的）信息，通常通過利用外部向量存儲和快速檢索實現。


## 1.3. 工具使用（Tool use）
- 代理學會調用外部API以獲取模型權重中缺少的額外信息，包括當前信息、代碼執行能力、訪問專有信息源等。

### 1.3.1. 組件一 : 規劃（Planning）
- **任務分解（Task Decomposition）**：通過思維鍊（Chain of Thought, CoT）和思維樹（Tree of Thoughts）等技術，將大任務轉化為多個可管理的小任務，並探索每一步的多種推理可能性。
- **自我反思（Self-Reflection）**：通過反思和行動（ReAct）和反思（Reflexion）等框架，代理能夠通過改進過去的行動決策和糾正先前的錯誤來迭代改進。

### 1.3.2. 組件二 : 記憶的（Memory）
記憶可以被定義為用於獲取、存儲、保留，以及稍後檢索信息的過程。在LLM應用中，我們可以將人腦中的記憶類型與LLM的功能進行類比：
- **感官記憶（Sensory Memory）**：做為學習原始輸入（包括文本、圖像或其他模態）的嵌入表示。這是記憶的最早階段，提供了在原始刺激結束後保留感官信息印象的能力。
- **短期記憶（Short-Term Memory）/ 工作記憶（Working Memory）**：做為上下文學習（in-context learning）。它是短暫且有限的，因為它受到 Transformer 有限上下文窗口長度的限制。短期記憶負責存儲我們當前意識到的信息，以及執行複雜認知任務（如學習和推理）所需的信息。
- **長期記憶（Long-Term Memory）**：做為代理在查詢時可以關注的外部向量存儲，可通過快速檢索訪問。長期記憶能夠存儲信息極長的時間，從幾天到數十年不等，具有基本無限的存儲容量。在 LLM 應用中，這相當於將信息的嵌入表示保存到一個向量存儲數據庫中，該數據庫支持快速最大內積搜索（Maximum Inner Product Search, MIPS）。

### 1.3.3. 組件三 : 工具使用（Tool Use）
- 工具使用是人類的一個顯著特點。配備 LLM 外部工具可以顯著擴展模型的能力。



# 3. 安裝與入門指南
## 3.1. 安裝 Conda
首先，安裝 Conda 環境管理器。推薦使用 Miniconda，因為它比 Anaconda 更輕量。可以從以下連結下載安裝：
[Miniconda](https://docs.anaconda.com/free/miniconda/index.html)


## 3.2. 建立 conda 環境
接著，使用以下命令建立一個新的 conda 環境並啟動他。此處以`XuanAgent`做為環境名稱，並安裝了 Python 3.10.9 版本。
```bash
conda create -n XuanAgent python=3.10.9
conda activate XuanAgent
```


## 3.3. 安裝 git 與 pytorch
透過以下命令在環境中安裝 Git 和 PyTorch。這裡安裝的是 PyTorch 2.0.1 版本，並確保相容於 CUDA 11.8。
P.S. 如果你需要安裝最新版本的 PyTorch，可以使用註解掉的命令行。
```bash
conda install -c anaconda git
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```


## 3.4. 下載 XuanAgent 專案，並安裝 requirements 中的套件
下載以下連結的專案，並置於根目錄底下：
[XuanAgent 專案](https://github.com/shiuan89910/XuanAgent/archive/refs/heads/main.zip)
>根目錄的位置
>Windows: C:\Users\使用者名稱
>Ubuntu: /home/使用者名稱

再透過以下命令進入專案目錄，此處為`XuanAgent`，並安裝所有依賴。
```bash
cd XuanAgent
pip install -r requirements.txt
```


## 3.5. 下載 LLM (GPTQ 量化模型)、Speech-to-Text (STT) 與 ocr-captcha 模型
關於開源 LLM (GPTQ 量化模型)、STT 與 ocr-captcha 模型的權重下載連結。您可以透過 Hugging Face 平台獲取這些資源，進行研究或開發工作。

### 3.5.1. 權重下載
開源 LLM (GPTQ 量化模型)、STT 與 ocr-captcha 模型的權重可以透過 [Hugging Face](https://huggingface.co/models) 進行下載。Hugging Face 提供了廣泛的預訓練模型，支持各種自然語言處理任務。

### 3.5.2. 授權條款注意事項
在使用本項目提供的開源 LLM (GPTQ 量化模型)、Speech-to-Text (STT) 與 ocr-captcha 模型或任何其他資源時，**強烈建議**用戶仔細查看每個模型或資源的授權條款。不同的模型和資源可能會有不同的授權要求，這可能會影響您使用這些資源的方式。
請前往相應的平台或資源頁面，如 [Hugging Face 模型庫](https://huggingface.co/models)，以獲取詳細的授權信息。確保您的使用方式符合這些授權條款的規定，以避免侵犯著作權或其他法律問題。
使用這些資源時，如果有任何疑問，建議咨詢法律專業人士或直接與模型/資源的提供者聯繫以獲取進一步的指導。

### P.S. 下載的模型請置於`XuanAgent`目錄底下


## 3.6. 啟動 .py 檔
### 3.6.1. llm_agent，執行以下命令
```bash
# 打開 llm_agent.py 檔，在此行 model_dir = os.path.join(root_path, "Mistral-7B-Instruct-v0.1-GPTQ") 的 "Mistral-7B-Instruct-v0.1-GPTQ" 輸入 LLM (GPTQ 量化模型) 的目錄名稱
# 透過 llm_agent.py 檔的 ga.run_agent("how many letters in the word educa") 使用單詞長度計算工具
# 透過 llm_agent.py 檔註解掉的 ga.run_agent("What is 2123 * 215123") 使用數學問題解答工具
# 透過 llm_agent.py 檔註解掉的 ga.run_agent("What is the current price of Microsoft stock") 使用股票價格查詢工具

python llm_agent.py
```

### 3.6.2. google_search_api，執行以下命令
```bash
# 透過此連結 https://serpapi.com/users/sign_in 註冊取得 API KEY
# 打開 google_search_api.py 檔，在此行 gsa = GoogleSearchAPI("api_key") 的 "api_key" 輸入 API KEY
# 打開 google_search_api.py 檔，在此行 ret = gsa.search("huggingface", **param_search) 的 "huggingface" 輸入要搜尋的關鍵字

python google_search_api.py
```

### 3.6.3. book_ticket，執行以下命令
```bash
# 打開 book_ticket.py 檔，設定 config = { ... } 內的設定值

python book_ticket.py

# 輸入 URL
```

### 3.6.4. gpt_api，執行以下命令
```bash
# 打開 gpt_api.py 檔，在此行 ga = GPTAPI("mail", "password") 的 "mail" 與 "password" 輸入 Google 登入帳號與密碼

python gpt_api.py

# 輸入 URL
```



# 4. 致謝
本專案的參考來源，特此致謝
[Lilian Weng's Blog](https://lilianweng.github.io/posts/2023-06-23-agent/)

[langchain-ai 的 langchain](https://github.com/langchain-ai/langchain)
