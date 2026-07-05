# Docker 기반 lab 사용 튜토리얼 (따라치기)

lab을 로컬 파이썬 대신 **Docker 컨테이너로** 실행해 `run → dataset-push → submit --pull`까지
가는 실습입니다. 개념·비-Docker 흐름은 [TUTORIAL.md](TUTORIAL.md), 이미지 빌드·배포·k8s는
[DOCKER.md](DOCKER.md)를 참고하세요 — 이 문서는 **컨테이너로 lab을 쓰는 법**에 집중합니다.

- 전제: **평가 서버가 이미 떠 있어야** 합니다(로컬/Docker/k8s 무관) → 서버 기동은 [DOCKER.md §1](DOCKER.md).
- GPU 없이(mock)도, GPU 실측(gemma)도 같은 이미지로 됩니다.

---

## 0. 준비 — 이미지 빌드 (1회)

```bash
cd ~/plr-prompt-lab
docker build -t plr-prompt-lab .        # CUDA llama-cpp 빌드 — 수 분 (GPU-free만 쓸 거면 그래도 필요)
```

---

## 1. ⚠ 가장 중요한 함정 — 컨테이너에서 서버 주소

컨테이너 안에서 **`localhost`는 서버가 아니라 그 컨테이너 자신**입니다. 서버가 호스트나
다른 노드에 있으면 반드시 **호스트/서버의 LAN IP**를 써야 합니다:

```bash
# ❌ 컨테이너 안에서 localhost → 자기 자신 (연결 실패)
# ✅ 서버의 실제 IP:포트
export SERVER=http://192.168.x.x:8890          # 서버 LAN IP (k8s면 LoadBalancer EXTERNAL-IP)
export TOKEN=<서버토큰>
```

> 서버도 같은 호스트의 도커면 `--network host`(리눅스)로 `localhost:8890` 접근도 가능하지만,
> **LAN IP 사용을 권장**합니다(k8s·다른 노드와 일관).

---

## 2. 공통 옵션 묶기

매 명령에 붙는 옵션이 기니 변수로 둡니다. **데이터셋은 호스트 `./datasets`를 볼륨 마운트**해
컨테이너의 `/app/datasets`에 연결 — 그래서 명령 안 경로는 `datasets/...` 그대로 씁니다.

```bash
cd ~/plr-prompt-lab      # 여기의 datasets/ 가 컨테이너와 공유됨

# GPU-free (mock)
LAB="docker run --rm \
  -e EVAL_SERVER_URL=$SERVER -e EVAL_SERVER_TOKEN=$TOKEN \
  -v $PWD/datasets:/app/datasets \
  plr-prompt-lab"

# GPU 실측 (gemma) — --gpus + HF_TOKEN + 모델 캐시 볼륨 추가
LABGPU="docker run --rm --gpus all \
  -e EVAL_SERVER_URL=$SERVER -e EVAL_SERVER_TOKEN=$TOKEN -e HF_TOKEN=<hf_...> \
  -v $PWD/datasets:/app/datasets -v plr_hf_cache:/hf-cache \
  plr-prompt-lab"
```

`plr-prompt-lab` 이미지의 ENTRYPOINT가 `lab.py`라, `$LAB` 뒤에 **서브명령만** 붙입니다.

---

## 3. GPU-free 실습 (mock)

### 3-1. 데이터셋 준비 (합성 demo)

```bash
$LAB demo --keep
# 컨테이너의 /app/datasets/demo = 호스트 ./datasets/demo 에 생성됨(볼륨 공유)
ls datasets/demo        # crops/ labels.jsonl manifest.yaml
```

### 3-2. run → push → submit --pull

```bash
$LAB run -X plr_v1.5_cot --dataset datasets/demo --model mock
$LAB dataset-push --dataset datasets/demo --name demo
$LAB submit --dataset demo --run-dir datasets/demo -X plr_v1.5_cot --pull
```

출력에 run id·지표가 뜨고, `--pull` 산출물이 **호스트에** 남습니다:

```bash
ls datasets/demo/pulled/     # gallery.html  metrics.json  report.html
xdg-open datasets/demo/pulled/report.html
# 리더보드: http://192.168.x.x:8890/d/demo (브라우저)
```

> mock은 고정 출력이라 정확도 숫자는 무의미하지만, **컨테이너 전체 흐름**이 도는지 확인용입니다.

---

## 4. 실측 (GPU, gemma)

`$LAB` 대신 `$LABGPU`를 쓰고 `--model gemma`로. 모델(Gemma-4-E4B GGUF)은 **첫 run 때
자동 다운로드**되어 `plr_hf_cache` 볼륨에 캐시됩니다(다음부터 빠름).

```bash
# datasets/mine 은 사람이 라벨한 실제 데이터셋(만드는 법: DATASET_GUIDE.md)
$LABGPU run -X plr_v1.5_cot --dataset datasets/mine --model gemma   # 첫 run: 모델 다운로드(수 GB)
$LABGPU dataset-push --dataset datasets/mine --name mine
$LABGPU submit --dataset mine --run-dir datasets/mine -X plr_v1.5_cot --pull
```

> 서버 흐름(push/submit)은 mock과 **동일** — 모델만 바뀝니다. `--gpus all`이 안 되면
> NVIDIA Container Toolkit 미설정([DOCKER.md §0](DOCKER.md)).

---

## 5. compose 방식 (선택)

`docker-compose.example.yml`의 `EVAL_SERVER_URL`·`EVAL_SERVER_TOKEN`·`HF_TOKEN`을 편집한 뒤:

```bash
docker compose -f docker-compose.example.yml run --rm lab \
  run -X plr_v1.5_cot --dataset datasets/mine --model gemma
docker compose -f docker-compose.example.yml run --rm lab \
  submit --dataset mine --run-dir datasets/mine -X plr_v1.5_cot --pull
```

---

## 6. 개선 루프 (버전 A/B)

```bash
# 프롬프트(prompts/<버전>/*.yaml) 수정 → 새 버전으로 run+submit
$LABGPU run    -X plr_v1.6_test --dataset datasets/mine --model gemma
$LABGPU submit --dataset mine --run-dir datasets/mine -X plr_v1.6_test
# 리더보드 http://<서버>:8890/d/mine 에 두 버전이 나란히 → Δ 비교
```

---

## 7. 트러블슈팅 (Docker 특화)

| 증상 | 원인 / 해결 |
|---|---|
| `서버 연결 실패` / `Connection refused` | 컨테이너에서 `localhost` 씀 → **서버 LAN IP**로. 또는 리눅스면 `--network host` |
| `--server 또는 EVAL_SERVER_URL 필요` | `$LAB`에 `-e EVAL_SERVER_URL` 누락 |
| `invalid or missing X-Auth-Token` | `EVAL_SERVER_TOKEN`이 서버와 다름 |
| 데이터셋을 컨테이너가 못 찾음 | `-v $PWD/datasets:/app/datasets` 마운트 + 명령은 `datasets/...` 상대경로로. `~/plr-prompt-lab`에서 실행 |
| `pulled/`가 호스트에 안 보임 | 볼륨 마운트 확인 — `datasets/<name>/pulled/`는 호스트 `./datasets/<name>/pulled/` |
| HF `401`/gated | `-e HF_TOKEN=<hf_...>` 유효값, 모델 접근 권한 |
| 첫 gemma run 매우 느림 | 모델 최초 다운로드(수 GB) — `plr_hf_cache` 볼륨에 캐시되어 다음부터 빠름 |
| `could not select device driver ... gpu` | NVIDIA Container Toolkit 미설정([DOCKER.md §0](DOCKER.md)) |
| `⚠ hash unverified` | 정상 동작. 이미지엔 `.git`이 없어 lab_sha는 비지만 surface_hash는 파일에서 계산돼 검증됨 |

---

## 8. 정리

```bash
rm -rf datasets/demo                 # 로컬 산출물(gitignore)
docker volume rm plr_hf_cache        # 모델 캐시 삭제(원하면)
```

> `datasets/`·크롭은 사적 CCTV — 이미지엔 `.dockerignore`로 안 들어가고, 볼륨으로만 공유됩니다.
