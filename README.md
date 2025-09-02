# dc_car_prompt_0919
Dacon Prompt 대회용 파일

## 평가지표

0.9 * 모델 분류 정확도 + 0.1 * 시스템 프롬프트 길이 점수

## 테스트 파일 프로세스
1. 주어진 예제파일 samples.csv를 불러와 'title', 'content'를 넣고 
2. openrouter api를 통해 'GPT_4o mini', temperature=0.4 에게 **시스템 프롬프트**를 변경하면서 
3. label=0 or 1을 반환하여 성적을 평가하고
4. 좋은 성적이 나올 때까지 반복