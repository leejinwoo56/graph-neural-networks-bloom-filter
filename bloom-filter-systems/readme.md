# bloom-filter-systems

Bloom Filter를 단일 머신 환경과 Spark 기반 분산 환경에서 직접 구현하고 성능을 비교한 프로젝트를 정리한 저장소입니다.  
단일 환경에서는 Bloom filter의 기본 동작과 false positive rate(FPR)를 실험하고, 분산 환경에서는 Spark RDD를 활용해 대규모 데이터셋에 대한 Bloom filter 생성, FPR 평가, 그리고 cascade Bloom filter와 dense Bloom filter의 비교 실험을 수행했습니다.  
이 프로젝트를 통해 확률적 자료구조의 동작 원리, 해시 기반 membership query, 그리고 분산 환경에서의 효율적인 필터 구성 방식을 구현 중심으로 익혔습니다.

## Projects

### 1. Single-Machine Bloom Filter Experiment

**File:** `src/run_bloom_filter_single.py`

#### 개요
단일 머신 환경에서 Bloom filter를 생성하고, 다양한 hash function 개수 `k`에 대해 empirical false positive rate(FPR)를 측정하는 실험 파이프라인입니다.

#### 구현 내용
- synthetic URL 데이터 생성
- train / negative test ID 집합 구성
- Bloom filter에 train URL 삽입
- negative URL query를 통한 false positive 개수 측정
- `k` 값을 바꿔가며 FPR 비교 실험 수행

#### 배운 점
- Bloom filter의 기본 동작 과정을 end-to-end로 구현하는 방법
- false positive rate가 hash function 개수와 어떻게 연결되는지 이해
- synthetic dataset을 활용해 membership query 구조를 실험하는 방법 학습
- 단일 환경에서 자료구조 구현과 실험 드라이버를 분리하는 코드 구조 경험

---

### 2. Single-Machine Bloom Filter Implementation

**File:** `src/bloom_filter_single.py`

#### 개요
NumPy 기반 bit array를 사용하여 Bloom filter의 핵심 연산인 insert와 query를 직접 구현한 프로젝트입니다.

#### 구현 내용
- 길이 `m`의 bit array 초기화
- key에 대한 다중 hash index 계산
- insert 연산을 통해 해당 위치 bit를 1로 설정
- query 연산을 통해 모든 bit가 1인지 확인
- membership query 결과를 boolean 형태로 반환

#### 배운 점
- Bloom filter가 정확한 저장이 아니라 확률적 membership test를 수행하는 구조임을 이해
- insert와 query가 동일한 hash 함수 집합을 공유해야 하는 이유 학습
- bit array 기반 자료구조가 메모리를 효율적으로 사용하는 방식 이해
- false positive는 발생할 수 있지만 false negative는 발생하지 않는다는 성질 이해

---

### 3. Bloom Hash Utilities

**File:** `src/bloom_hash_utils.py`

#### 개요
SHA-256 기반 해시를 활용하여, Bloom filter에서 사용할 `k`개의 hash index를 생성하는 유틸리티 모듈입니다.

#### 구현 내용
- seed를 포함한 deterministic hash 생성
- key 문자열을 byte 형태로 변환
- seed를 바꿔가며 여러 hash 값 생성
- hash 값을 Bloom filter bit array 범위 `[0, m)` 로 변환
- `k`개의 index 리스트 반환

#### 배운 점
- Bloom filter에서 여러 hash 함수를 구현할 때 seed variation을 활용하는 방법
- 해시 함수가 deterministic해야 하는 이유 이해
- 문자열 key를 안정적으로 index 공간에 매핑하는 방법 학습
- 해시 분포 품질이 Bloom filter 성능에 영향을 준다는 점 이해

---

### 4. Spark Bloom Filter Experiment

**File:** `src/run_bloom_filter_spark.py`

#### 개요
Spark 기반 분산 환경에서 Bloom filter를 생성하고, `k` sweep 실험과 cascade Bloom filter vs dense Bloom filter 비교 실험을 수행하는 실행 파이프라인입니다.

#### 구현 내용
- Spark session 생성
- train / negative test RDD 생성
- distributed Bloom filter 생성
- distributed false positive rate 측정
- 여러 `k` 값에 대한 FPR 비교
- two-stage cascade Bloom filter와 dense Bloom filter 비교 실험 수행

#### 배운 점
- Bloom filter 실험을 Spark 기반으로 확장하는 방법
- 단일 머신 실험과 분산 실험의 구조 차이 이해
- 분산 환경에서 데이터 생성, 캐싱, 반복 실험을 구성하는 방법 학습
- cascade 구조가 false positive rate에 어떤 영향을 줄 수 있는지 실험적으로 이해

---

### 5. Distributed Bloom Filter Implementation

**File:** `src/bloom_filter_spark.py`

#### 개요
Spark RDD를 이용해 분산 Bloom filter를 생성하고, distributed membership query 및 two-stage cascade Bloom filter 평가를 수행하는 프로젝트입니다.

#### 구현 내용
- key를 hash index들로 변환한 뒤 `(index, 1)` 형태로 전개
- `reduceByKey`를 이용한 bit aggregation
- NumPy bit array로 최종 Bloom filter 구성
- broadcast bit array를 사용한 distributed query 수행
- negative query 집합에 대해 distributed FPR 계산
- two-stage cascade Bloom filter 구성 및 평가

#### 배운 점
- Bloom filter 생성 과정을 map-reduce 스타일로 분산화하는 방식 이해
- Spark의 `flatMap`, `reduceByKey`, `broadcast` 가 이런 문제에 어떻게 활용되는지 학습
- 분산 환경에서 query 연산 시 shared state를 broadcast하는 이유 이해
- 단일 Bloom filter와 cascade Bloom filter의 구조적 차이를 구현 관점에서 경험

---

## Skills Demonstrated

- Python
- NumPy
- PySpark
- Bloom Filter
- Probabilistic Data Structures
- Hash Functions
- False Positive Rate (FPR)
- Distributed Computing
- Spark RDD
- Broadcast Variables
- ReduceByKey
- Membership Query
- Algorithm Implementation
