# TODO Record

## Function TODO
- [X] 不同語言的評估流程。
- [X] 評估兩個不同但相關的基準的難度流程。
    - [X] 記錄：
        - [X] 每個主題的難度計數和總和
        - [X] 難度變異數
    - [X] 可以直接使用現成的 pipeline 修改就行 (需要 inference_model)
        - [X] 更新：可以套在現成的 pipeline.

- [X] 翻譯品質評估流程。
    - 可以直接使用現成的 pipeline 修改就行（需要提供路徑）
- [X] 根據 deepseek 的論文，修正 inference prompt

## Benchmark TODO
- [X] Flores-200
- [ ] MBPP

## Template TODO
- [X] 新增不同的 template 在以下檔案:
    1. [trans_template.py](../src/BenchWeaver/eval/template/trans/trans_template.py)
    2. [translation_prompt.json](..//prompt/translation_prompt.json)
- [X] 翻譯目標:
    - [X] 正確地將源語言翻譯成目標語言。
    - [X] 對齊源語言和目標語言的難度。
    - [X] 專有名詞正確翻譯。

## Exepriment TODO
- [ ] 現有的平行語料 (P-MMEval) 試比較
- [ ] 同類型 Benchmark 難度差異比較
- [X] 不同翻譯 Template 比較（Maybe 翻譯精準度、模型提升比較）

## 需要解決的問題
- [X] 問題是 messsage list, 需要完整的翻譯，可能不適合直接打包成 str 直翻 (不能確定番後格式會不會跑掉，尤其是長文本)
    - 可能解法: few shot 的先翻，因此只需要翻譯最後一個 user prompt 欄位？ (更新：已修改完成，一起翻譯，翻完再重排回去)
- [ ] reference benchmark 的 subject 可能選擇相關的會更好，而不是 random
    - 可能要等 Benchmark 都加完成後再看看

