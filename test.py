
import collections
import json
import math
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

class CodeCorruptionDetector:
    """檢測程式碼是否因重複內容而損壞的類別"""
    
    def __init__(self):
        # 可調整的閾值參數
        self.max_consecutive_repeat = 50  # 連續重複字元的最大數量
        self.entropy_threshold = 2.5      # 熵的最小閾值
        self.repetitive_pattern_threshold = 0.7  # 重複模式比例閾值
        self.min_code_length = 50         # 最小程式碼長度才進行檢測
    
    def detect_corruption(self, code_string: str, passed: bool, verbose: bool = True) -> dict:
        """
        檢測程式碼是否損壞
        
        Args:
            code_string: 待檢查的程式碼字串
            verbose: 是否輸出詳細信息
            
        Returns:
            dict: 包含檢測結果的字典
        """
        result = {
            'is_corrupted': False,
            'corruption_reasons': [],
            'details': {},
            'confidence': 0.0
        }
        # 如果程式碼已經通過檢測，則不進行損壞檢測
        if passed:
            result['is_corrupted'] = False
            result['confidence'] = 0.0
            if verbose:
                print("程式碼已通過檢測，無需進行損壞檢測。")
            return result
        
        if len(code_string) < self.min_code_length:
            result['details']['length'] = len(code_string)
            if verbose:
                print(f"程式碼長度太短 ({len(code_string)} 字元)，跳過檢測。")
            return result
        
        corruption_indicators = 0
        max_indicators = 4
        
        # 1. 檢查連續重複字元
        consecutive_repeat = self._check_consecutive_repeats(code_string)
        result['details']['max_consecutive_repeats'] = consecutive_repeat
        
        if consecutive_repeat >= self.max_consecutive_repeat:
            result['corruption_reasons'].append(f"檢測到 {consecutive_repeat} 個連續重複字元")
            corruption_indicators += 1
            if verbose:
                print(f"⚠️  發現 {consecutive_repeat} 個連續重複字元")
        
        # 2. 檢查熵值
        entropy = self._calculate_entropy(code_string)
        result['details']['entropy'] = entropy
        
        if entropy < self.entropy_threshold:
            result['corruption_reasons'].append(f"熵值過低 ({entropy:.2f} < {self.entropy_threshold})")
            corruption_indicators += 1
            if verbose:
                print(f"⚠️  熵值過低: {entropy:.2f}")
        
        # 3. 檢查重複模式
        repetitive_ratio = self._check_repetitive_patterns(code_string)
        result['details']['repetitive_pattern_ratio'] = repetitive_ratio
        
        if repetitive_ratio > self.repetitive_pattern_threshold:
            result['corruption_reasons'].append(f"重複模式比例過高 ({repetitive_ratio:.2f})")
            corruption_indicators += 1
            if verbose:
                print(f"⚠️  重複模式比例過高: {repetitive_ratio:.2f}")
        
        # 4. 檢查異常字符分佈
        char_anomaly = self._check_character_anomalies(code_string)
        result['details']['character_anomaly_score'] = char_anomaly
        
        if char_anomaly > 0.8:
            result['corruption_reasons'].append(f"字符分佈異常 (分數: {char_anomaly:.2f})")
            corruption_indicators += 1
            if verbose:
                print(f"⚠️  字符分佈異常，分數: {char_anomaly:.2f}")
        
        # 計算整體信心度和結果
        result['confidence'] = corruption_indicators / max_indicators
        result['is_corrupted'] = corruption_indicators >= 2  # 至少兩個指標異常才判斷為損壞
        
        if verbose:
            print(f"\n📊 檢測結果:")
            print(f"   損壞可能性: {'高' if result['is_corrupted'] else '低'}")
            print(f"   信心度: {result['confidence']:.2f}")
            print(f"   異常指標數: {corruption_indicators}/{max_indicators}")
        
        return result
    
    def _check_consecutive_repeats(self, code_string: str) -> int:
        """檢查最大連續重複字元數"""
        if len(code_string) <= 1:
            return 0
        
        max_repeat = 0
        current_repeat = 0
        
        for i in range(1, len(code_string)):
            if code_string[i] == code_string[i-1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 0
        
        return max_repeat
    
    def _calculate_entropy(self, code_string: str) -> float:
        """計算字串的熵值"""
        if not code_string:
            return 0.0
        
        frequencies = collections.Counter(code_string)
        total_chars = len(code_string)
        
        entropy = 0.0
        for freq in frequencies.values():
            probability = freq / total_chars
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _check_repetitive_patterns(self, code_string: str) -> float:
        """檢查重複模式的比例"""
        # 移除空白字符進行分析
        cleaned_code = re.sub(r'\s+', '', code_string)
        if len(cleaned_code) < 20:
            return 0.0
        
        # 查找重複的子字串
        repetitive_chars = 0
        
        # 檢查長度為2-10的重複模式
        for pattern_length in range(2, min(11, len(cleaned_code) // 4)):
            for i in range(len(cleaned_code) - pattern_length * 2):
                pattern = cleaned_code[i:i + pattern_length]
                # 計算此模式在後續文本中的重複次數
                remaining_text = cleaned_code[i + pattern_length:]
                repeats = 0
                pos = 0
                
                while pos <= len(remaining_text) - pattern_length:
                    if remaining_text[pos:pos + pattern_length] == pattern:
                        repeats += 1
                        pos += pattern_length
                    else:
                        pos += 1
                
                if repeats >= 3:  # 至少重複3次才算異常
                    repetitive_chars += pattern_length * (repeats + 1)
        
        return min(1.0, repetitive_chars / len(cleaned_code))
    
    def _check_character_anomalies(self, code_string: str) -> float:
        """檢查字符分佈異常"""
        if not code_string:
            return 0.0
        
        char_freq = collections.Counter(code_string)
        total_chars = len(code_string)
        
        # 檢查是否有某個字符佔比過高
        max_char_ratio = max(freq / total_chars for freq in char_freq.values())
        
        # 檢查非常見程式語言字符的比例
        common_code_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_(){}[];.,=+-*/<>!&|"\'#$%^@`~: \t\n\r')
        unusual_chars = sum(1 for char in code_string if char not in common_code_chars)
        unusual_ratio = unusual_chars / total_chars if total_chars > 0 else 0
        
        # 綜合異常分數
        anomaly_score = max_char_ratio * 0.7 + unusual_ratio * 0.3
        
        return anomaly_score


def check_code_corruption(code_input: str, passed: bool, verbose: bool = True) -> bool:
    """
    簡化的檢查函數
    
    Args:
        code_input: 待檢查的程式碼字串
        verbose: 是否顯示詳細信息
    
    Returns:
        bool: True 表示可能損壞，False 表示正常
    """
    detector = CodeCorruptionDetector()
    result = detector.detect_corruption(code_input, passed, verbose)
    return result['is_corrupted']


def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of dictionaries.
    
    Args:
        file_path (str): Path to the JSONL file.
        
    Returns:
        list: List of dictionaries loaded from the JSONL file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def process_entry(entry):
        code_snippet = entry.get('completion', '')
        passed = entry.get('passed', False)
        return check_code_corruption(code_snippet, passed, verbose=False)

def mp_check_file(file_path: str, verbose: bool = True) -> None:
    # given a file path, use mp method as below, and print the result
    print("=======================================================")
    print(f"Processing file: {file_path}")
    # Load test data
    test_data = load_jsonl(test_data_path)
    # Use multiprocessing to process entries in parallel
    with Pool(processes=cpu_count() - 4) as pool:
        results = list(tqdm(pool.imap(process_entry, test_data), total=len(test_data), desc="Processing entries"))

    # Filter out corrupted codes
    corrupted_codes = [entry for entry, is_corrupted in zip(test_data, results) if is_corrupted]
    # print total data
    print(f"Total entries processed: {len(test_data)}")
    print(f"Total passed codes: {sum(1 for entry in test_data if entry.get('passed', True))}")
    print(f"Total corrupted codes found: {len(corrupted_codes)}")
    print(f"Total compile errors found: {len(test_data) - sum(1 for entry in test_data if entry.get('passed', True)) - len(corrupted_codes)}")

    # Save corrupted codes to a file
    output_path = "corrupted_codes.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for code in corrupted_codes:
            f.write(json.dumps(code, ensure_ascii=False) + '\n')
        print(f"Corrupted codes saved to: {output_path}")
    print("=======================================================")
    
if __name__ == "__main__":
    test_data_path_list = [
        "/work/u5110390/BenchWeaver/score/pmmeval_exp/en/humaneval-xl/en/mxeval_results.jsonl",
        "/work/u5110390/BenchWeaver/score/pmmeval_exp/en/humaneval-xl/ko/mxeval_results.jsonl",
        "/work/u5110390/BenchWeaver/score/pmmeval_exp/en/humaneval-xl/zh/mxeval_results.jsonl",
        "/work/u5110390/BenchWeaver/score/pmmeval_exp/ko/humaneval-xl/en/mxeval_results.jsonl",
        "/work/u5110390/BenchWeaver/score/pmmeval_exp/ko/humaneval-xl/ko/mxeval_results.jsonl",
        "/work/u5110390/BenchWeaver/score/pmmeval_exp/ko/humaneval-xl/zh/mxeval_results.jsonl",
        "/work/u5110390/BenchWeaver/score/pmmeval_exp/zh/humaneval-xl/en/mxeval_results.jsonl",
        "/work/u5110390/BenchWeaver/score/pmmeval_exp/zh/humaneval-xl/ko/mxeval_results.jsonl",
        "/work/u5110390/BenchWeaver/score/pmmeval_exp/zh/humaneval-xl/zh/mxeval_results.jsonl",
    ]
    for test_data_path in test_data_path_list:
        mp_check_file(test_data_path, verbose=True)
        print()

    