import logging
import os
import transformers
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.chains import LLMMathChain
from langchain.llms import HuggingFacePipeline
from langchain.tools import BaseTool
from langchain.tools.render import render_text_description_and_args
from langchain.tools.render import render_text_description
from pathlib import Path
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, TextGenerationPipeline
from typing import Optional, Type


proj_name = "XuanAgent"
try:
    user_name = os.getlogin()
    root_path = f"C:\\Users\\{user_name}\\{proj_name}"
except:
    root_path = f"/{proj_name}"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GPTAgent():
    def __init__(self, **kwargs):
        """初始化 GPTAgent 類別，設定模型參數"""
        self.llm = self._init_open_llm(**kwargs)

    def _init_open_llm(self, **kwargs):
        """初始化開源語言模型"""
        try:
            model_dir = kwargs["model_dir"]
            path_to_model = Path(model_dir)
            pt_path = None
            for ext in [".safetensors", ".pt", ".bin"]:
                found = list(path_to_model.glob(f"*{ext}"))
                if len(found) > 0:
                    pt_path = found[-1]
                    break
            use_safetensors = pt_path.suffix == ".safetensors"
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=kwargs["use_fast"], trust_remote_code=kwargs["trust_remote_code"])
            model = AutoGPTQForCausalLM.from_quantized(
                path_to_model, 
                model_basename=pt_path.stem, 
                use_safetensors=use_safetensors,
                device=kwargs["device"],
                inject_fused_attention=kwargs["inject_fused_attention"],
                disable_exllama=kwargs["disable_exllama"],
                trust_remote_code=kwargs["trust_remote_code"],
                )
            generate_text = transformers.pipeline(
                model=model, tokenizer=tokenizer,
                return_full_text=kwargs["return_full_text"],
                task=kwargs["task"],
                temperature=kwargs["temperature"],
                max_new_tokens=kwargs["max_new_tokens"],
                repetition_penalty=kwargs["repetition_penalty"],
                )
            return HuggingFacePipeline(pipeline=generate_text)
        except Exception as e:
            logging.error(f"_init_open_llm() ERR: {e}")
            return None
        
    def get_llm(self):
        """獲取語言模型"""
        return self.llm
        
    def init_agent(self, **kwargs):
        """初始化代理"""
        try:
            prompt = hub.pull(f'hwchase17/{kwargs["hub"]}')
            prompt = prompt.partial(
                #tools=render_text_description_and_args(tools), # "hub": "react-multi-input-json",
                tools=render_text_description(kwargs["tools"]),
                tool_names=", ".join([t.name for t in kwargs["tools"]]))
            llm_with_stop = self.llm.bind(stop=["\nObservation"])
            agent = {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"])  
                } | prompt | llm_with_stop | ReActSingleInputOutputParser()
                #} | prompt | llm_with_stop | ReActJsonSingleInputOutputParser() # "hub": "react-json",
                #} | prompt | llm_with_stop | JSONAgentOutputParser() # "hub": "react-multi-input-json",
            self.agent = AgentExecutor(
                agent=agent, 
                tools=kwargs["tools"],
                return_intermediate_steps=kwargs["return_intermediate_steps"],
                handle_parsing_errors=kwargs["handle_parsing_errors"],
                max_execution_time=kwargs["max_execution_time"],
                max_iterations=kwargs["max_iterations"],    
                verbose=kwargs["verbose"],
                )
        except Exception as e:
            logging.error(f"init_agent() ERR:{e}")
            self.agent =  None
        
    def run_agent(self, input):
        """運行代理"""
        self.agent.run(input=input)


def main():
    try:
        # 定義模型與創建代理實例
        model_dir = os.path.join(root_path, "Mistral-7B-Instruct-v0.1-GPTQ")
        param_open_llm = {
            "model_dir": model_dir,
            "use_fast": True,
            "trust_remote_code": True,
            "device": "cuda:0",
            "inject_fused_attention": False,
            "disable_exllama": True,
            "return_full_text": True,
            "task": "text-generation",
            "temperature": 0.0,
            "max_new_tokens": 512,
            "repetition_penalty": 1.1,
            }
        ga = GPTAgent(**param_open_llm)

        # 獲取股票價格的 Tool
        class CurrentStockPriceInput(BaseModel):
            ticker: str = Field(description="Ticker symbol of the stock")

        class CurrentStockPriceTool(BaseTool):
            name = "get_current_stock_price"
            description = """
                Useful when you want to get current stock price.
                You should enter the stock ticker symbol recognized by the yahoo finance
                """
            args_schema: Type[BaseModel] = CurrentStockPriceInput
            return_direct=False
            handle_tool_error=True
            def _run(
                self, 
                ticker: str,
                run_manager: Optional[CallbackManagerForToolRun] = None,
                ) -> str:
                price_response = get_current_stock_price(ticker)
                return price_response

            def _arun(
                self, 
                ticker: str,
                run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
                ) -> str:
                raise NotImplementedError("get_current_stock_price does not support async")
    
        def get_current_stock_price(ticker):
            return {"price": 334.57000732421875, "currency": "USD"}
    
        stock_tool = CurrentStockPriceTool(
            callbacks=[HumanApprovalCallbackHandler()]
            )

        # 獲取單詞長度的 Tool
        def get_word_length(word: str) -> int:
            return len(word)
    
        word_length_tool = Tool(
            name="get_word_length",
            func=get_word_length,
            description="Returns the length of a word.",
            #callbacks=[HumanApprovalCallbackHandler()],
            )
        
        # 數學計算的 Tool
        math_tool = Tool(
            name="Calculator",
            func=LLMMathChain.from_llm(llm=ga.get_llm(), verbose=True).run,
            description="useful for when you need to answer questions about math",
            #callbacks=[HumanApprovalCallbackHandler()],
            )

        # 定義與初始化，並運行代理
        param_agent = {
            "hub": "react", # "react" or "react-json" or "react-multi-input-json"
            "tools": [stock_tool, word_length_tool, math_tool],
            "return_intermediate_steps": True,
            "handle_parsing_errors": True,
            "max_execution_time": 1,
            "max_iterations": 1,
            "verbose": True,
            }
        ga.init_agent(**param_agent)
        ga.run_agent("how many letters in the word educa")
        #ga.run_agent("What is 2123 * 215123")
        #ga.run_agent("What is the current price of Microsoft stock")
    except Exception as e:
        logging.error(f"main() ERR: {e}")


if __name__ == "__main__":
    main()