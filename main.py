import json
import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TranslationProcessor:
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        
        # 检查必要的环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未设置 OPENAI_API_KEY 环境变量")
        
        api_base = os.getenv("OPENAI_API_BASE")
        if not api_base:
            logger.warning("未设置 OPENAI_API_BASE，使用默认值")
            api_base = "https://api.openai.com/v1"
        
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        logger.info(f"初始化完成. 模型: {self.model}, API Base: {api_base}")
        
        self.system_prompt = """你是一个专业的游戏本地化翻译助手。我会给你一个包含英文文本数组的 JSON，请将这些文本翻译成地道的简体中文。

        规则：
        1. 保持所有特殊标记完全不变：
        - 换行符 \n
        - 变量占位符 {{xxx}}
        - 图标标识符 :xxx:
        - 标点符号
        - Markdown标记 如 *text* 或 **text**
        2. 保持游戏原有的语气和风格
        3. 确保游戏术语翻译统一
        4. 翻译要自然流畅，符合中文表达习惯

        请以 JSON 格式返回翻译结果，保持原有顺序。

        输入格式示例：
        {
            "texts": [
                "Text with {{variable}}",
                "Text with *markdown* and **bold**"
            ]
        }

        输出格式示例：
        {
            "translations": [
                "带有{{variable}}的文本",
                "带有*markdown*和**加粗**的文本"
            ]
        }"""
        self.processed_items = {}
        self.errors = []

    def create_batches(self, data: Dict[str, str], batch_size: int = 10) -> List[Tuple[List[str], List[str]]]:
        """将数据分成批次，返回 (keys, values) 的元组列表"""
        items = list(data.items())
        batches = []
        
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            # 分离键和值
            keys, values = zip(*batch_items)
            batches.append((list(keys), list(values)))
            
        logger.info(f"将{len(items)}个条目分成了{len(batches)}个批次")
        return batches

    def translate_batch(self, batch: Tuple[List[str], List[str]]) -> Dict[str, str]:
        """使用 OpenAI API 翻译一个批次"""
        keys, values = batch
        try:
            logger.info(f"正在翻译批次，包含 {len(values)} 个条目")
            logger.debug(f"批次第一个条目: {values[0]}")
            
            # 构建要发送给 LLM 的 JSON 格式文本
            translation_request = {"texts": values}
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(translation_request, ensure_ascii=False)}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}  # 确保返回 JSON 格式
            )
            
            response_content = response.choices[0].message.content
            logger.debug(f"API 响应: {response_content}")
            
            # 解析 JSON 响应
            try:
                translated_data = json.loads(response_content)
                if not isinstance(translated_data, dict) or "translations" not in translated_data:
                    raise ValueError("Invalid response format")
                
                translations = translated_data["translations"]
                if len(translations) != len(keys):
                    raise ValueError(f"Translation count mismatch: got {len(translations)}, expected {len(keys)}")
                
                # 将翻译结果与原始键配对
                translated = dict(zip(keys, translations))
                logger.info(f"成功解析翻译结果，包含 {len(translated)} 个条目")
                return translated
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析错误: {e}")
                logger.error(f"原始响应: {response_content}")
                return None
                
        except Exception as e:
            error_msg = f"翻译错误: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None

    def process_file(self, input_file: str, output_file: str, test_mode: bool = True):
        """处理整个翻译文件"""
        try:
            logger.info(f"开始处理文件: {input_file}")
            
            # 读取输入文件
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功读取输入文件，包含 {len(data)} 个条目")

            # 创建批次
            batches = self.create_batches(data, batch_size=10)
            
            # 测试模式只处理前两个批次
            if test_mode:
                batches = batches[:2]
                logger.info("测试模式: 只处理前两个批次")

            # 处理每个批次
            for i, batch in enumerate(batches):
                logger.info(f"正在处理批次 {i+1}/{len(batches)}")
                
                translated_batch = self.translate_batch(batch)
                if translated_batch:
                    self.processed_items.update(translated_batch)

            # 保存结果
            output_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_items": len(self.processed_items),
                    "errors": len(self.errors)
                },
                "translations": self.processed_items,
                "errors": self.errors
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"处理完成。成功翻译: {len(self.processed_items)}, 错误: {len(self.errors)}")
            if self.errors:
                logger.info("错误列表:")
                for error in self.errors:
                    logger.error(f"- {error}")

        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}", exc_info=True)

def main():
    logger.info("开始运行翻译程序")
    processor = TranslationProcessor()
    processor.process_file(
        input_file='base-translations.json',
        output_file='zh-translations.json',
        test_mode=False
    )

if __name__ == "__main__":
    main()