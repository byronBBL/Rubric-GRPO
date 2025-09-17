#!/usr/bin/env python3
"""
Rubric Reward Service

这个服务使用 vLLM 加载 reward model，对收到的数据进行评分。
数据格式: {rubric: rubric, question: question, response: response}
返回格式: [{score: ..., weight: ...}, ...]
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from flask import Flask, request, jsonify
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RubricCriterion:
    """单个评价标准"""
    description: str
    weight: int


@dataclass
class EvaluationRequest:
    """评价请求"""
    rubric: List[Dict[str, Any]]
    question: str
    response: str


@dataclass
class EvaluationResult:
    """评价结果 - 包含RubricCriterion信息和评分"""
    description: str
    weight: int
    score: float


class RubricRewardService:
    """Rubric 奖励服务"""
    
    def __init__(
        self,
        model_path: str = ".../Qwen3-32B",
        sys_prompt_path: str = ".../bbl_reward/prompts/eva_resp_with_rubric_sys_new.txt",
        user_prompt_path: str = ".../bbl_reward/prompts/eva_resp_with_rubric_user_new.txt",
        likert_sys_prompt_path: str = ".../bbl_reward/prompts/eva_likert_sys.txt",
        likert_user_prompt_path: str = ".../bbl_reward/prompts/eva_likert_users.txt",
        outcome_sys_prompt_path: str = ".../bbl_reward/prompts/eva_outcome_sys.txt",
        outcome_user_prompt_path: str = ".../bbl_reward/prompts/eva_outcome_user.txt",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 2
    ):
        """
        初始化 Rubric Reward Service
        
        Args:
            model_path: 模型路径
            sys_prompt_path: 系统提示文件路径
            user_prompt_path: 用户提示文件路径
            likert_sys_prompt_path: Likert评分系统提示文件路径
            likert_user_prompt_path: Likert评分用户提示文件路径
            gpu_memory_utilization: GPU内存使用率
            tensor_parallel_size: 张量并行大小
        """
        self.model_path = model_path
        self.sys_prompt_path = sys_prompt_path
        self.user_prompt_path = user_prompt_path
        self.likert_sys_prompt_path = likert_sys_prompt_path
        self.likert_user_prompt_path = likert_user_prompt_path
        self.outcome_sys_prompt_path = outcome_sys_prompt_path
        self.outcome_user_prompt_path = outcome_user_prompt_path
        
        # 加载提示模板
        self.system_prompt = self._load_prompt_template(sys_prompt_path)
        self.user_prompt_template = self._load_prompt_template(user_prompt_path)
        
        # 加载 Likert 评分提示模板
        self.likert_system_prompt = self._load_prompt_template(likert_sys_prompt_path)
        self.likert_user_prompt_template = self._load_prompt_template(likert_user_prompt_path)

        # 加载 Outcome 评分提示模板
        self.outcome_system_prompt = self._load_prompt_template(outcome_sys_prompt_path)
        self.outcome_user_prompt_template = self._load_prompt_template(outcome_user_prompt_path)
        
        # 初始化 tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 初始化 vLLM
        logger.info(f"Loading model from {model_path}")
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )
        
        # 采样参数
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=16
        )
        
        logger.info("RubricRewardService initialized successfully")
    
    def _load_prompt_template(self, file_path: str) -> str:
        """加载提示模板"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error loading prompt template from {file_path}: {e}")
            raise
    
    def _parse_rubric_criteria(self, rubric: List[Dict[str, Any]]) -> List[RubricCriterion]:
        """解析 rubric 标准"""
        criteria = []
        for item in rubric:
            try:
                criterion = RubricCriterion(
                    description=item.get("description", ""),
                    weight=int(item.get("weight", 1))
                )
                criteria.append(criterion)
            except Exception as e:
                logger.warning(f"Error parsing rubric item {item}: {e}")
                print()
        return criteria
    
    def _create_single_prompt(
        self,
        question: str,
        response: str,
        criterion: RubricCriterion
    ) -> List[Dict[str, str]]:
        """为单个标准创建提示"""
        # 构建 single_rubric_criterion
        single_rubric_criterion = f"{criterion.description}"
        
        # 填充用户提示模板，使用新的格式化语法
        user_prompt = self.user_prompt_template.format(
            prompt=question,
            response=response,
            single_rubric_criterion=single_rubric_criterion
        )
        
        # 构建对话格式
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    
    def _extract_rating_from_response(self, response_text: str) -> Optional[float]:
        """从模型响应中提取评分"""
        try:
            # 尝试匹配 JSON 块
            json_pattern = r'```json\s*\n?(.*?)\n?```'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    data = json.loads(json_str)
                    rating = data.get("rating")
                    if rating is not None:
                        return float(rating)
                except json.JSONDecodeError:
                    pass
            
            # 备用方案：仅匹配 0 或 1
            rating_pattern = r'"rating"\s*:\s*([01])'
            rating_match = re.search(rating_pattern, response_text)
            if rating_match:
                return float(rating_match.group(1))
                
        except Exception as e:
            logger.warning(f"Error extracting rating from response: {e}")
        
        logger.warning(f"Could not extract rating from response: {response_text[:200]}...")
        return None
    
    def _format_messages_for_vllm(self, messages: List[Dict[str, str]]) -> str:
        """将消息格式化为 vLLM 可接受的格式"""
        formatted = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        return formatted
    
    def evaluate_batch(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """批量评价"""
        # try:
        # 解析 rubric 标准
        criteria = self._parse_rubric_criteria(request.rubric)
        if not criteria:
            logger.error("No valid criteria found in rubric")
            return []
        
        # 为每个标准创建提示
        prompts = []
        for criterion in criteria:
            messages = self._create_single_prompt(
                request.question,
                request.response,
                criterion
            )
            formatted_prompt = self._format_messages_for_vllm(messages)
            prompts.append(formatted_prompt)
        
        logger.info(f"Processing {len(prompts)} evaluation prompts")
        
        # 批量生成
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # 处理结果
        results = []
        for i, output in enumerate(outputs):
            criterion = criteria[i]
            response_text = output.outputs[0].text
            
            # 提取评分
            rating = self._extract_rating_from_response(response_text)
            
            if rating is not None:
                # 确保评分在 1-10 范围内
                rating = max(0, min(1, rating))
                result = EvaluationResult(
                    description=criterion.description,
                    weight=criterion.weight,
                    score=rating
                )
                results.append(result)
                logger.info(f"des '{criterion.description}': score={rating}, weight={criterion.weight}")
            else:
                # 默认分数
                logger.warning(f"Failed to extract rating for criterion '{criterion.description}', using default score 1.0")
                result = EvaluationResult(
                    description=criterion.description,
                    weight=criterion.weight,
                    score=0
                )
                results.append(result)
        
        return results
            
        # except Exception as e:
        #     logger.error(f"Error in evaluate_batch: {e}")
        #     return []
    
    def process_request(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理单个请求
        
        Args:
            data: 格式为 {rubric: rubric, question: question, response: response}
            
        Returns:
            List[Dict]: 格式为 [{description: ..., weight: ..., score: ...}, ...]
        """
        # try:
        request = EvaluationRequest(
            rubric=data.get("rubric", []),
            question=data.get("question", ""),
            response=data.get("response", "")
        )
        
        results = self.evaluate_batch(request)
        
        # 转换为字典格式，包含完整的RubricCriterion信息和score
        return [
            {
                "description": result.description, 
                "weight": result.weight,
                "score": result.score
            } 
            for result in results
            ]
            
        # except Exception as e:
        #     logger.error(f"Error processing request: {e}")
        #     return []

    def process_batch_fast(self, requests_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        快速批量处理多个请求 - 一次性推理所有prompts
        
        Args:
            requests_list: 请求列表，每个请求格式为 {rubric: rubric, question: question, response: response}
            
        Returns:
            List[Dict]: 批量结果，格式与batch_evaluate_api相同
        """
        print(f"开始快速批量处理 {len(requests_list)} 个请求...")
        
        # 收集所有prompts和对应的映射信息
        all_prompts = []
        prompt_mappings = []  # 记录每个prompt对应的request_idx和criterion_idx
        
        for req_idx, req_data in enumerate(requests_list):
            # 解析当前请求的criteria
            request = EvaluationRequest(
                rubric=req_data.get("rubric", []),
                question=req_data.get("question", ""),
                response=req_data.get("response", "")
            )
            
            criteria = self._parse_rubric_criteria(request.rubric)
            if not criteria:
                continue
            
            # 为每个标准创建prompt
            for criterion_idx, criterion in enumerate(criteria):
                messages = self._create_single_prompt(
                    request.question,
                    request.response,
                    criterion
                )
                formatted_prompt = self._format_messages_for_vllm(messages)
                all_prompts.append(formatted_prompt)
                
                # 记录映射关系
                prompt_mappings.append({
                    'request_idx': req_idx,
                    'criterion_idx': criterion_idx,
                    'criterion': criterion
                })
        
        print(f"总共生成 {len(all_prompts)} 个prompts")
        
        # 一次性批量推理所有prompts
        if all_prompts:
            outputs = self.llm.generate(all_prompts, self.sampling_params)
            print(f"vLLM生成了 {len(outputs)} 个结果")
        else:
            outputs = []
        
        # 按request重新组织结果
        results_by_request = {}
        
        for prompt_idx, (output, mapping) in enumerate(zip(outputs, prompt_mappings)):
            request_idx = mapping['request_idx']
            criterion_idx = mapping['criterion_idx']
            criterion = mapping['criterion']
            
            if request_idx not in results_by_request:
                results_by_request[request_idx] = []
            
            # 提取评分
            response_text = output.outputs[0].text
            rating = self._extract_rating_from_response(response_text)
            
            if rating is not None:
                rating = max(0, min(1, rating))
            else:
                logger.warning(f"Failed to extract rating for criterion '{criterion.description}', using default score 5.0")
                rating = 0
            
            # 添加结果
            result = {
                "description": criterion.description,
                "weight": criterion.weight,
                "score": rating
            }
            results_by_request[request_idx].append(result)
        
        # 构建最终返回格式
        final_results = []
        for req_idx in range(len(requests_list)):
            if req_idx in results_by_request:
                results = results_by_request[req_idx]
                
                # 计算统计信息
                total_score = sum(result["score"] * result["weight"] for result in results)
                # 分母计算：正的权重*10 + 负的权重*1
                total_weight = sum(result["weight"] * 1.0 for result in results)
                weighted_average = total_score / total_weight if total_weight > 0 else 0.0
                
                # 计算 score_list (每个 rubric 的 score * weight)
                score_list = [float(result["score"]) * float(result["weight"]) for result in results]
                # 将 score_list 序列化为字符串，避免 numpy 数组转换错误
                import json
                score_list_json = json.dumps(score_list)
                
                final_results.append({
                    "index": req_idx,
                    "results": results,
                    "score_list": score_list_json,
                    "summary": {
                        "weighted_average": weighted_average
                    }
                })
            else:
                # 没有有效criteria的请求
                final_results.append({
                    "index": req_idx,
                    "error": "无有效的评价标准"
                })
        
        print(f"快速批量处理完成，返回 {len(final_results)} 个结果")
        return final_results
            
        # except Exception as e:
        #     logger.error(f"快速批量处理失败: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     return []

    def _create_likert_prompt(
        self,
        question: str,
        reference_answer: str,
        response: str
    ) -> List[Dict[str, str]]:
        """为 Likert 评分创建提示"""
        # 填充用户提示模板
        user_prompt = self.likert_user_prompt_template.format(
            prompt=question,
            reference=reference_answer,
            response=response
        )
        
        # 构建对话格式
        messages = [
            {"role": "system", "content": self.likert_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages

    def _create_outcome_prompt(
        self,
        question: str,
        groundtruth: str,
        response: str
    ) -> List[Dict[str, str]]:
        """为 Outcome 0/1 评分创建提示"""
        # 基本字段校验：任一为空直接报错
        empty_fields = []
        if not question:
            empty_fields.append("question")
        if not groundtruth:
            empty_fields.append("groundtruth")
        if not response:
            empty_fields.append("response")
        if empty_fields:
            raise ValueError(f"Outcome评估所需字段为空: {empty_fields}")

        user_prompt = self.outcome_user_prompt_template.format(
            prompt=question,
            groundtruth=groundtruth,
            response=response
        )

        messages = [
            {"role": "system", "content": self.outcome_system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return messages

    def process_outcome_batch_fast(self, requests_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        快速批量处理 Outcome 0/1 评分请求
        
        Args:
            requests_list: 列表，每项包含 {groundtruth: str, question: str, response: str}
        Returns:
            List[Dict]: 形如 [{"index": i, "score": float}, ...]，分数范围[0,1]
        """
        print(f"开始快速批量处理 {len(requests_list)} 个 Outcome 评分请求...")

        all_prompts = []

        for req_idx, req_data in enumerate(requests_list):
            question = req_data.get("question", "")
            groundtruth = req_data.get("groundtruth", "")
            response = req_data.get("response", "")

            messages = self._create_outcome_prompt(question, groundtruth, response)
            formatted_prompt = self._format_messages_for_vllm(messages)
            all_prompts.append(formatted_prompt)

        print(f"总共生成 {len(all_prompts)} 个 Outcome 评分 prompts")
        print(f"example prompt: {all_prompts[0]}")

        if all_prompts:
            outputs = self.llm.generate(all_prompts, self.sampling_params)
            print(f"vLLM生成了 {len(outputs)} 个结果")
        else:
            outputs = []

        final_results = []
        for req_idx, output in enumerate(outputs):
            response_text = output.outputs[0].text
            rating = self._extract_rating_from_response(response_text)

            if rating is not None:
                rating = max(0.0, min(1.0, rating))
            else:
                logger.warning(f"Failed to extract Outcome rating for request {req_idx}, using default score 0.0")
                rating = 0.0

            final_results.append({
                "index": req_idx,
                "score": rating
            })

        print(f"快速 Outcome 批量处理完成，返回 {len(final_results)} 个结果")
        return final_results

    def process_likert_batch_fast(self, requests_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        快速批量处理 Likert 评分请求
        
        Args:
            requests_list: 请求列表，每个请求格式为 {reference_answer: str, question: str, response: str}
            
        Returns:
            List[Dict]: 批量结果，格式为 [{"index": i, "score": float}, ...]
        """
        print(f"开始快速批量处理 {len(requests_list)} 个 Likert 评分请求...")
        
        # 收集所有prompts
        all_prompts = []
        
        for req_idx, req_data in enumerate(requests_list):
            question = req_data.get("question", "")
            reference_answer = req_data.get("reference_answer", "")
            response = req_data.get("response", "")
            
            # 创建 Likert 评分提示
            messages = self._create_likert_prompt(question, reference_answer, response)
            formatted_prompt = self._format_messages_for_vllm(messages)
            all_prompts.append(formatted_prompt)
        
        print(f"总共生成 {len(all_prompts)} 个 Likert 评分 prompts")
        
        # 一次性批量推理所有prompts
        if all_prompts:
            outputs = self.llm.generate(all_prompts, self.sampling_params)
            print(f"vLLM生成了 {len(outputs)} 个结果")
        else:
            outputs = []
        
        # 处理结果
        final_results = []
        
        for req_idx, output in enumerate(outputs):
            # 提取评分
            response_text = output.outputs[0].text
            rating = self._extract_rating_from_response(response_text)
            
            if rating is not None:
                rating = max(1.0, min(10.0, rating))
            else:
                logger.warning(f"Failed to extract Likert rating for request {req_idx}, using default score 5.0")
                rating = 5.0
            
            final_results.append({
                "index": req_idx,
                "score": rating
            })
        
        print(f"快速 Likert 批量处理完成，返回 {len(final_results)} 个结果")
        return final_results


# 全局服务实例
service = None

def init_service():
    """初始化服务"""
    global service
    logger.info("🚀 正在初始化 Rubric Reward Service...")
    service = RubricRewardService()
    logger.info("✅ 服务初始化完成")

# 创建Flask应用
app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate_api():
    """评价API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "缺少JSON数据"}), 400
        
        # 验证必要字段
        required_keys = ['rubric', 'question', 'response']
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            return jsonify({"error": f"缺少必要字段: {missing_keys}"}), 400
        
        # 处理请求
        results = service.process_request(data)
        
        if results:
            return jsonify(results)
        else:
            return jsonify({"error": "评价失败"}), 500
            
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_evaluate', methods=['POST'])
def batch_evaluate_api():
    """批量评价API - 串行处理"""
    try:
        data = request.get_json()
        if not data or 'requests' not in data:
            return jsonify({"error": "缺少requests字段"}), 400
        
        requests_list = data['requests']
        results_list = []
        
        print(f"开始处理 {len(requests_list)} 个批量评价请求...")
        
        for i, req_data in tqdm(enumerate(requests_list), total=len(requests_list), desc="处理batch请求"):
            # 验证每个请求的必要字段
            required_keys = ['rubric', 'question', 'response']
            missing_keys = [key for key in required_keys if key not in req_data]
            
            if missing_keys:
                results_list.append({
                    "index": i,
                    "error": f"缺少必要字段: {missing_keys}"
                })
                continue
            
            # 处理请求
            results = service.process_request(req_data)
            
            if results:
                # 计算统计信息
                total_score = sum(result["score"] * result["weight"] for result in results)
                total_weight = sum(result["weight"] * (10 if result["weight"] > 0 else 1) for result in results)
                weighted_average = total_score / total_weight if total_weight > 0 else 0.0
                
                # 计算 score_list (每个 rubric 的 score * weight)
                score_list = [float(result["score"]) * float(result["weight"]) for result in results]
                # 将 score_list 序列化为字符串，避免 numpy 数组转换错误
                import json
                score_list_json = json.dumps(score_list)
                
                results_list.append({
                    "index": i,
                    "results": results,
                    "score_list": score_list_json,
                    "summary": {
                        "weighted_average": weighted_average
                    }
                })
            else:
                results_list.append({
                    "index": i,
                    "error": "评价失败"
                })
        
        return jsonify({"batch_results": results_list})
        
    except Exception as e:
        logger.error(f"处理批量请求时出错: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/batch_evaluate_fast', methods=['POST'])
def batch_evaluate_fast_api():
    """快速批量评价API - 并行处理"""
    try:
        data = request.get_json()
        if not data or 'requests' not in data:
            return jsonify({"error": "缺少requests字段"}), 400
        
        requests_list = data['requests']
        print(f"开始快速批量处理 {len(requests_list)} 个请求...")
        
        # 验证所有请求的必要字段
        valid_requests = []
        request_indices = []
        
        for i, req_data in enumerate(requests_list):
            required_keys = ['rubric', 'question', 'response']
            missing_keys = [key for key in required_keys if key not in req_data]
            
            if missing_keys:
                # 跳过无效请求，稍后填充错误结果
                continue
            else:
                valid_requests.append(req_data)
                request_indices.append(i)
        
        print(f"有效请求数量: {len(valid_requests)}")
        
        if not valid_requests:
            return jsonify({"error": "没有有效的请求"}), 400
        
        # 使用新的批量处理方法
        batch_results = service.process_batch_fast(valid_requests)
        
        # 构建最终结果列表
        results_list = [None] * len(requests_list)
        
        # 填充有效请求的结果
        for result_idx, request_idx in enumerate(request_indices):
            if result_idx < len(batch_results):
                results_list[request_idx] = batch_results[result_idx]
        
        # 填充无效请求的错误信息
        for i, req_data in enumerate(requests_list):
            if results_list[i] is None:
                required_keys = ['rubric', 'question', 'response']
                missing_keys = [key for key in required_keys if key not in req_data]
                results_list[i] = {
                    "index": i,
                    "error": f"缺少必要字段: {missing_keys}"
                }
        
        return jsonify({"batch_results": results_list})
        
    except Exception as e:
        logger.error(f"快速批量处理请求时出错: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/likert_evaluate_fast', methods=['POST'])
def likert_evaluate_fast_api():
    """快速批量 Likert 评分API"""
    try:
        data = request.get_json()
        if not data or 'requests' not in data:
            return jsonify({"error": "缺少requests字段"}), 400
        
        requests_list = data['requests']
        print(f"开始快速批量处理 {len(requests_list)} 个 Likert 评分请求...")
        
        # 验证所有请求的必要字段
        valid_requests = []
        request_indices = []
        
        for i, req_data in enumerate(requests_list):
            required_keys = ['reference_answer', 'question', 'response']
            missing_keys = [key for key in required_keys if key not in req_data]
            
            if missing_keys:
                # 跳过无效请求，稍后填充错误结果
                continue
            else:
                valid_requests.append(req_data)
                request_indices.append(i)
        
        print(f"有效 Likert 评分请求数量: {len(valid_requests)}")
        
        if not valid_requests:
            return jsonify({"error": "没有有效的请求"}), 400
        
        # 使用新的 Likert 批量处理方法
        batch_results = service.process_likert_batch_fast(valid_requests)
        
        # 构建最终结果列表
        results_list = [None] * len(requests_list)
        
        # 填充有效请求的结果
        for result_idx, request_idx in enumerate(request_indices):
            if result_idx < len(batch_results):
                results_list[request_idx] = batch_results[result_idx]
        
        # 填充无效请求的错误信息
        for i, req_data in enumerate(requests_list):
            if results_list[i] is None:
                required_keys = ['reference_answer', 'question', 'response']
                missing_keys = [key for key in required_keys if key not in req_data]
                results_list[i] = {
                    "index": i,
                    "error": f"缺少必要字段: {missing_keys}"
                }
        
        return jsonify({"batch_results": results_list})
        
    except Exception as e:
        logger.error(f"快速 Likert 批量处理请求时出错: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/outcome_evaluate_fast', methods=['POST'])
def outcome_evaluate_fast_api():
    """快速批量 Outcome 评分API"""
    try:
        data = request.get_json()
        if not data or 'requests' not in data:
            return jsonify({"error": "缺少requests字段"}), 400
        
        requests_list = data['requests']
        print(f"开始快速批量处理 {len(requests_list)} 个 Outcome 评分请求...")
        
        # 验证所有请求的必要字段
        valid_requests = []
        request_indices = []
        
        for i, req_data in enumerate(requests_list):
            required_keys = ['groundtruth', 'question', 'response']
            missing_or_empty = [key for key in required_keys if (key not in req_data or not req_data.get(key))]
            
            if missing_or_empty:
                # 跳过无效请求，稍后填充错误结果
                continue
            else:
                valid_requests.append(req_data)
                request_indices.append(i)
        
        print(f"有效 Outcome 评分请求数量: {len(valid_requests)}")
        
        if not valid_requests:
            return jsonify({"error": "没有有效的请求"}), 400
        
        # 使用 Outcome 批量处理方法
        batch_results = service.process_outcome_batch_fast(valid_requests)
        
        # 构建最终结果列表
        results_list = [None] * len(requests_list)
        
        # 填充有效请求的结果
        for result_idx, request_idx in enumerate(request_indices):
            if result_idx < len(batch_results):
                results_list[request_idx] = batch_results[result_idx]
        
        # 填充无效请求的错误信息
        for i, req_data in enumerate(requests_list):
            if results_list[i] is None:
                required_keys = ['groundtruth', 'question', 'response']
                missing_or_empty = [key for key in required_keys if (key not in req_data or not req_data.get(key))]
                results_list[i] = {
                    "index": i,
                    "error": f"缺少或为空的必要字段: {missing_or_empty}"
                }
        
        return jsonify({"batch_results": results_list})
        
    except Exception as e:
        logger.error(f"快速 Outcome 批量处理请求时出错: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_api():
    """健康检查"""
    try:
        if service is None:
            return jsonify({"status": "error", "message": "服务未初始化"}), 503
        
        return jsonify({
            "status": "healthy",
            "service": "rubric_reward_service",
            "backend": "vllm",
            "model_path": service.model_path
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """根路径"""
    return jsonify({
        "service": "Rubric Reward Service API",
        "version": "1.0.0",
        "endpoints": {
            "/evaluate": "POST - 单次评价",
            "/batch_evaluate": "POST - 批量评价 (串行处理)", 
            "/batch_evaluate_fast": "POST - 快速批量评价 (并行处理)",
            "/likert_evaluate_fast": "POST - 快速批量 Likert 评分",
            "/outcome_evaluate_fast": "POST - 快速批量 Outcome 0/1 评分",
            "/health": "GET - 健康检查"
        }
    })

def main():
    """主函数 - 启动Flask服务"""
    # 初始化服务
    init_service()
    
    # 启动Flask服务
    logger.info("🌐 启动Flask服务在 http://0.0.0.0:8003")
    logger.info("📖 API文档:")
    logger.info("  POST /evaluate - 单次评价")
    logger.info("  POST /batch_evaluate - 批量评价 (串行处理)")
    logger.info("  POST /batch_evaluate_fast - 快速批量评价 (并行处理)")
    logger.info("  POST /likert_evaluate_fast - 快速批量 Likert 评分")
    logger.info("  GET /health - 健康检查")
    
    app.run(host='0.0.0.0', port=8003, debug=False, threaded=True)

if __name__ == "__main__":
    main()
