# 데이터 관리 모듈 (Data Management Module)

이 모듈은 반려동물 건강 모니터링 시스템의 데이터 관리 및 공유 기능을 담당합니다. 블록체인 기술을 통해 안전하고 투명한 데이터 관리 및 공유를 가능하게 합니다.

## 주요 구성 요소

### 1. 데이터 저장 관리자 (DataStorageManager)

반려동물 프로필, 센서 데이터, 건강 기록, 케어 플랜 등의 데이터를 안전하게 저장하고 관리하는 컴포넌트입니다.

- 데이터 암호화 및 안전한 저장
- 데이터 버전 관리
- 백업 및 복원 기능
- 데이터 검색 및 필터링

### 2. 블록체인 데이터 공유 (BlockchainDataSharing)

블록체인 기술을 활용한 안전하고 투명한 데이터 공유 시스템입니다.

- 데이터 무결성 보장
- 데이터 접근 권한 관리
- 데이터 접근 이력 추적
- 권한 부여 및 회수

### 3. 데이터 교환 (SharedDataExchange)

암호화된 데이터 교환 기능을 제공합니다.

- 암호화된 데이터 패키지 생성
- 데이터 무결성 검증
- 안전한 데이터 교환 프로토콜

## 블록체인 구현

본 모듈에 구현된 블록체인은 다음 특징을 가집니다:

1. **Proof of Work**: 블록 생성 시 난이도 조정 가능한 작업 증명 알고리즘 사용
2. **데이터 무결성**: SHA-256 해시 알고리즘을 통한 데이터 무결성 보장
3. **투명한 이력 관리**: 모든 데이터 접근과 공유 이력이 블록체인에 기록
4. **분산 저장**: 개별 블록 파일로 저장되어 데이터 복원성 강화

## 사용 예시

### 1. 데이터 관리 모듈 초기화

```python
from data_management_module import DataStorageManager, BlockchainDataSharing, SharedDataExchange

# 데이터 저장 관리자 초기화
storage_manager = DataStorageManager("./data")

# 블록체인 데이터 공유 초기화
blockchain_sharing = BlockchainDataSharing("./blockchain_data")

# 데이터 교환 초기화
data_exchange = SharedDataExchange(blockchain_sharing)
```

### 2. 데이터 공유

```python
# 수의사와 데이터 공유
share_result = blockchain_sharing.share_pet_data(
    pet_id="pet12345",
    recipient_id="vet98765",
    data_categories=["basic_profile", "medical_records"],
    expiration_time=time.time() + 7 * 24 * 60 * 60  # 7일 후 만료
)

print(f"Share ID: {share_result['share_id']}")
```

### 3. 접근 권한 검증

```python
# 접근 권한 검증
validation = blockchain_sharing.validate_access_permission(
    "pet12345",
    "vet98765",
    "medical_records"
)

if validation['has_access']:
    print("Access granted")
else:
    print("Access denied")
```

### 4. 접근 이력 로깅

```python
# 데이터 접근 이력 로깅
blockchain_sharing.log_data_access(
    "pet12345",
    "vet98765",
    "medical_records",
    "read",
    "success"
)
```

### 5. 데이터 무결성 검증

```python
# 데이터 무결성 검증
verification = blockchain_sharing.verify_data_integrity(
    "pet12345",
    data_hash
)

if verification['verified']:
    print("Data integrity verified")
else:
    print("Data integrity check failed")
```

## 주의사항

- 블록체인 데이터는 정기적으로 백업하는 것이 좋습니다.
- 고성능 환경에서는 블록 마이닝 난이도를 조정할 수 있습니다.
- 민감한 데이터는 항상 암호화하여 저장하고 공유해야 합니다.

## 기술적 배경

이 모듈은 Ucaretron Inc.의 특허 기술을 기반으로 구현되었습니다. 블록체인 기술을 활용하여 데이터의 무결성을 보장하고, 안전한 데이터 공유를 가능하게 합니다.
