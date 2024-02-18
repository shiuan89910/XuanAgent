# 0. 目錄


# 1. XuanAgent 專案簡介
透過大型語言模型智能代理，用於加強 AI 系統的決策能力，提升代理的學習效率和適應性。

- 大型語言模型（Large Language Model, LLM）驅動的自主代理系統
本文概述了 LLM 驅動的自主代理系統的關鍵組件和功能。更多詳細信息，請參考[Lilian Weng's Blog](https://lilianweng.github.io/posts/2023-06-23-agent/)。

在一個 LLM 驅動的自主代理（Autonomous Agent）系統中，LLM 充當代理的大腦，並由幾個關鍵組件：
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



# 3. 致謝
本專案的參考來源，特此致謝
