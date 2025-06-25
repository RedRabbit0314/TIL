"""
Structured Logprobs, LLM 토큰의 로그 확률에 대한 정보를 제공하는 오픈소스 라이브러리

- LLM의 구조화된 출력의 신뢰도를 분석하고, JSON 스키마를 사용해 모델이 일관된 형식의 응답을 생성하도록 보장.

Structured Logprobs이 제공하는 핵심 기능
    1. 토큰 로그 확률 분석: 각 응답의 신뢰도를 평가하기 위해 토큰 로그 확률 데이터를 제공
    2. 로그 확률 추가 방법:
        - add_logprobs: 로그 확률 데이터를 응답 필드로 별도 추가
        - add_logprobs_inline: 메시지 내용 내부에 로그 확률 포함
    3. 문자-토큰 매핑 기능
        - map_characters_to_token_indices: 응답 내 문자와 토큰 인덱스를 매핑
"""

import math
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types import ResponseFormatJSONSchema
from structured_logprobs.main import add_logprobs, add_logprobs_inline

# Load Config
current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(current_path, "..", ".."))
env_path = os.path.abspath(os.path.join(parent_path, ".env"))

load_dotenv(env_path)


def sample_structured_logprobs():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": ("대한민국을 대표하는 대표 축구선수는?"),
            }
        ],
        logprobs=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        },
    )

    chat_completion = add_logprobs(completion)
    print("=" * 30)
    prob = math.exp(int(chat_completion.log_probs[0]["name"]))  # 확률로 변환
    percent = prob * 100  # 퍼센트 변환
    print(f"답변 정답 확률: {percent:4f}%")
    print("=" * 30)
    chat_completion_inline = add_logprobs_inline(completion)
    print("=" * 30)
    print(
        (chat_completion_inline.choices[0].message.content)
        .encode()
        .decode("unicode_escape")
    )
    print("=" * 30)
    return chat_completion


if __name__ == "__main__":
    sample_structured_logprobs()
    """
    예시 답변:
    ==============================
    답변 정답 확률: 100.000000%
    ==============================
    ==============================
    {"name": "손흥민", "name_logprob": -0.012841579809901305}
    ==============================
    """
