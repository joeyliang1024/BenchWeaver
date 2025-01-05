from vllm import LLM, SamplingParams

def main():
    model_name = "microsoft/Phi-3.5-mini-instruct"
    max_model_len = 8192 
    max_new_tokens = 8192 
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    # Load the model and tokenizer
    llm = LLM(model=model_name, max_model_len=max_model_len)
    tokenizer = llm.get_tokenizer()
    FULL_NAME_PROMPT = """[文章開始]{text}[文章結束]\n\n請針對以上內容中的所有縮寫或代名詞替換為其全名或完整形式，根據上下文展開每個縮寫，並將代名詞替換為具體的主語或指代對象。請只輸出修改後的文章，不要輸出其他的內容。"""
    text = """法務部函\n中華民國93年12月24日\n法統字第0931503776號 主 旨：檢送本部主動公開資訊目錄一份如附件，詳細內容刊載於法務部全球資訊網（www.moj.gov.tw）。請查照。\n說 明：依據行政資訊公開辦法第七條規定刊登行政院公報。\n部 長　陳定南\n法務部主動公開資訊\n資訊種類\n內容要旨\n作成或取得時間\n保管期間\n保管場所\n業務統計\n法務統計摘要\n九十三年十二月十五日\n一年\n法務部統計處"""
    print(text)
    messages = [
        {"role": "user", "content": FULL_NAME_PROMPT.format_map({'text':text})},
    ]

    output = llm.generate(tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False), sampling_params=sampling_params)
    print(output[0].outputs[0].text)

if __name__ == "__main__":
    main()