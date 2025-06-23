import argparse
import os
from typing import Tuple
from pathlib import Path

import torch
from accelerate import disk_offload
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Load Config
current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(current_path, "..", ".."))
env_path = os.path.abspath(os.path.join(parent_path, ".env"))

load_dotenv(env_path)

# Huggingface login
login(token=os.getenv("hf_token"), add_to_git_credential=True)


def model_definition(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        device_map="cpu",  # 전체를 CPU에 로드
        low_cpu_mem_usage=True,  # 메모리 절약
        offload_folder="offload",  # 디스크 오프로딩 폴더
        offload_state_dict=True,  # state_dict 도 디스크 기반 로딩
    )

    return tokenizer, model


# Preprocess Knowledge
def preprocess_knowledge(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str
) -> DynamicCache:
    """
    Prepare Knowledge kv cache for CAG.
    Args:
        model: HuggingFace model with automatic device mapping
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess. which is basically a prompt
    Returns:
        DynamicCache: KV Cache
    """
    embed_device = model.model.embed_tokens.weight.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
    past_key_values = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    return outputs.past_key_values


# Prepare the knowledges kvcache
def prepare_kvcache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    documents_path: str,
    answer_instruction: str = None,
):
    if answer_instruction is None:
        answer_instruction = "Answer the question with a super short answer."

    with open(documents_path, "r") as f:
        documents = f.read()

    knowledges = """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an legal assistant for giving short answers based on given reports. <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {documents}
    ------------------------------------------------
    {answer_instruction}
    Question:
    """

    data = {"documents": documents, "answer_instruction": answer_instruction}
    knowledges = knowledges.format(**data)

    # Get the knowledge cache
    kv = preprocess_knowledge(model, tokenizer, knowledges)
    kv_len = kv.key_cache[0].shape[-2]
    print("kvlen: ", kv_len)
    return kv, kv_len


# Query Answering
def clean_up(kv: DynamicCache, origin_len: int):
    """
    Truncate the KV Cache to the original length.
    """
    for i in range(len(kv.key_cache)):
        kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]


def generate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    past_key_values,
    max_new_tokens: int = 300,
) -> torch.Tensor:
    """
    Generate text with greedy decoding.

    Args:
        model: HuggingFace model with automatic device mapping.
        input_ids: Input token ids
        past_key_values: KV Cache for Knowledge
        max_new_tokens: Maximum new tokens to generate
    """

    embed_device = model.model.embed_tokens.weight.device

    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids

    # eos_token_id 안전 처리
    eos_token_ids = model.config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=next_token, past_key_values=past_key_values, use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            next_token = next_token.to(embed_device)

            past_key_values = outputs.past_key_values
            output_ids = torch.cat([output_ids, next_token], dim=1)

            # EOS 조건 처리 (배치 단위로도 대응 가능)
            if next_token[0].item() in eos_token_ids and step > 0:
                break

    return output_ids[:, origin_ids.shape[-1] :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model definition")
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        action="store",  # 인자를 받았을 때 어떤 방식으로 처리할지를 지정하는 옶션
        metavar="Model Name",  # 도움말에 표시할 인자 이름 변경
        help="모델 정의 (meta-llama/Meta-Llama-3-8B etc.)",  # 인자 설명 (자동 -h 도움말에 표시됨)
        required=True,
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        action="store",
        metavar="User Query",
        help="사용자 요청 질문",
        required=True,
    )

    parser.add_argument(
        "-k",
        "--knowledge_path",
        type=str,
        action="store",
        metavar="Required Knowledge",
        help="지식 베이스가 존재하는 파일 경로",
        required=True,
    )

    args = parser.parse_args()
    tokenizer, model = model_definition(args.model_id)
    knowledge_cache, kv_len = prepare_kvcache(
        model=model, tokenizer=tokenizer, documents_path=args.knowledge_path
    )
    clean_up(knowledge_cache, kv_len)
    input_ids = tokenizer.encode(args.query, return_tensors="pt").to(model.device)
    output = generate(model, input_ids, knowledge_cache)
    generated_text = tokenizer.decode(
        output[0], skip_special_tokens=True, temperature=None
    )

    print(f"Response of the model:\n {generated_text}")
