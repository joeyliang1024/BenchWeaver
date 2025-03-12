# TODO Record

## Function TODO
- [ ] 不同語言的評估流程。
- [ ] 評估兩個不同但相關的基準的難度流程。
    - [ ] 記錄：
        - [ ] 每個主題的難度計數和總和
        - [ ] 難度變異數

## Template TODO
- [ ] 新增不同的 template 在以下檔案:
    1. [trans_template.py](../src/BenchWeaver/eval/template/trans/trans_template.py)
    2. [translation_prompt.json](..//prompt/translation_prompt.json)
- [ ] 翻譯目標:
    - [ ] 正確地將源語言翻譯成目標語言。
    - [ ] 對齊源語言和目標語言的難度。
    - [ ] 專有名詞正確翻譯。

## Exepriment TODO
- [ ] 現有的平行語料 (P-MMEval) 試比較
- [ ] 同類型 Benchmark 難度差異比較
- [ ] 不同翻譯 Template 比較（Maybe 翻譯精準度、模型提升比較）
- [ ] 是否有跨語言的測試結果

## 需要解決的問題
- [X] 問題是 messsage list, 需要完整的翻譯，可能不適合直接打包成 str 直翻 (不能確定番後格式會不會跑掉，尤其是長文本)
    - 可能解法: few shot 的先翻，因此只需要翻譯最後一個 user prompt 欄位？ (更新：已修改完成，一起翻譯，翻完再重排回去)
- [ ] reference benchmark 的 subject 可能選擇相關的會更好，而不是 random
    - 可能要等 Benchmark 都加完成後再看看