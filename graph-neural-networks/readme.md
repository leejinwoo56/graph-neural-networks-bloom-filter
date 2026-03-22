# graphsage-node-classification

GraphSAGE 기반 그래프 신경망을 직접 구현하며 학습한 프로젝트를 정리한 저장소입니다.  
NumPy와 SciPy만을 사용하여 2-layer GraphSAGE 모델의 forward pass, loss 계산, macro-F1 평가, backpropagation, weight update를 직접 구현하고, PubMed citation network 데이터셋에 적용해 node classification을 수행했습니다.  
이 프로젝트를 통해 그래프 구조를 반영한 표현 학습, message passing 기반 aggregation, 그리고 그래프 분류 문제의 학습 과정을 구현 중심으로 익혔습니다.
## Projects

### 1. GraphSAGE PubMed Training Pipeline

**File:** `src/train_graphsage_pubmed.py`

#### 개요
PubMed citation network 데이터를 불러오고, GraphSAGE 모델을 학습한 뒤 train, validation, test 성능을 출력하는 전체 실행 파이프라인입니다.

#### 구현 내용
- PubMed 데이터셋 로드
- 입력 feature 차원과 클래스 수 추출
- 2-layer GraphSAGE 모델 생성
- epoch 단위 gradient descent 학습 수행
- train / validation loss, accuracy, macro-F1 출력
- 최종 test 성능 평가

#### 배운 점
- 그래프 신경망 학습 파이프라인의 전체 흐름 이해
- train / validation / test 분할의 역할과 필요성 학습
- accuracy뿐 아니라 macro-F1 같은 분류 평가 지표를 함께 보는 이유 이해
- 모델 구현과 실행 스크립트를 분리하는 코드 구조 경험

---

### 2. GraphSAGE Model Implementation

**File:** `src/graphsage_model.py`

#### 개요
NumPy와 SciPy만을 사용하여 mean aggregation과 concatenation 기반의 2-layer GraphSAGE 모델을 직접 구현한 프로젝트입니다.

#### 구현 내용
- row-normalized adjacency를 이용한 이웃 평균 aggregation
- self embedding과 neighbor embedding의 concatenation
- 2-layer GraphSAGE forward pass 구현
- softmax 기반 다중 클래스 분류 출력
- cross-entropy loss 계산
- accuracy 및 macro-F1 계산
- backpropagation을 통한 가중치 gradient 계산
- SGD 기반 weight update

#### 배운 점
- GraphSAGE의 message passing 구조를 직접 구현하며 이해
- GNN에서 이웃 정보 aggregation이 어떤 역할을 하는지 학습
- forward propagation과 backward propagation을 그래프 신경망에 적용하는 방식 이해
- 딥러닝 프레임워크 없이도 GNN의 핵심 연산을 직접 구현해볼 수 있다는 점 경험

---

### 3. PubMed Dataset Loader

**File:** `src/pubmed_dataset_loader.py`

#### 개요
전처리된 PubMed citation graph 데이터를 adjacency matrix, feature matrix, label, 그리고 train/validation/test mask 형태로 불러오는 데이터 로더 유틸리티입니다.

#### 구현 내용
- sparse adjacency matrix 로드
- sparse node feature matrix 로드
- one-hot label 및 integer label 로드
- train / validation / test boolean mask 로드
- GraphSAGE 학습에 필요한 데이터 포맷으로 반환

#### 배운 점
- 그래프 데이터셋이 adjacency, feature, label, mask로 구성되는 방식 이해
- sparse matrix를 활용해 그래프 데이터를 효율적으로 다루는 방법 학습
- 모델 코드와 데이터 로딩 코드를 분리하는 구조의 장점 이해
- node classification 문제에서 split mask가 어떻게 사용되는지 학습

---

## Skills Demonstrated

- Python
- NumPy
- SciPy
- Graph Neural Networks
- GraphSAGE
- Node Classification
- Message Passing
- Mean Aggregation
- Softmax Classification
- Cross-Entropy Loss
- Backpropagation
- Macro-F1 Evaluation
- Sparse Matrix Processing
- Algorithm Implementation
