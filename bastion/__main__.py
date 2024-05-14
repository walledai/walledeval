from eval import WMDP
from guard import ClaudeGuard
from llm import HF_LLM

if __name__ == "__main__":
    print("Loading Phi-3")
    llm = HF_LLM("microsoft/Phi-3-mini-4k-instruct")
    print("Phi-3 successfully loaded")
    
    key = input("Insert Claude API Key: ")
    
    print("Loading Claude 3")
    guard = ClaudeGuard(api_key = key)
    print("Claude 3 successfully loaded")
    
    print("Loading WMDP Benchmark")
    wmdp = WMDP()
    print("WMDP Benchmark successfully loaded\n\n")
    
    wmdp.test(llm, guard)