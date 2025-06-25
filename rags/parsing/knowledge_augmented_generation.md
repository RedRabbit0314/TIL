## KAG(Knowledge-Augmented Generation): 논리적 추론 및 정보 검색 프레임워크

- KAG 는 도메인 지식 기반의 논리적 추론 및 질문-답변을 위한 프레임워크

- 정확하고 신뢰할 수 있는 도메인 지식 활용을 가능하게 함
cf) 
    1. 전통적인 RAG의 모호성: 벡터 유사성 계산 방식의 한계를 극복
    2. GraphRAG의 잡음 문제: OpenIE(Open Information Extraction) 방식의 잡음을 최소화.

## KAG의 주요 특징
- KAG 는 DIKW 계층(Data, Information, Knowledge, Wisdom) 을 참조하여 SPG 를 LLM 친화적인 버전으로 업그레이드함.

![지식 표현 이미지](https://discuss.pytorch.kr/uploads/default/original/2X/4/41728c3be97d2904880da1af27a7e42a411064ff.png)

- KAG 프레임워크는 KG-Builder, KG-Solver 및 KAG-Model 의 세 가지 주요 구성 요소로 이루어져있으며, 현재 KAG-Model 만 공개되지 않았음.


