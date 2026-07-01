# config.py
import os

# REST API 설정
RESTAPI_HOST = os.getenv("RESTAPI_HOST", "ziosummary-api")
RESTAPI_PORT = int(os.getenv("RESTAPI_PORT", "3000"))
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "token")

# Redis 설정
REDIS_HOST = os.getenv("REDIS_HOST", "ziosummary-redis-stream")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Redis Stream 설정
REDIS_STREAM_REQUEST = os.getenv("REDIS_STREAM_REQUEST", "zs:stream:ir:request")  # 요청 수신 스트림
REDIS_STREAM_EVENTS = os.getenv("REDIS_STREAM_EVENTS", "zs:stream:ir:events")     # 이벤트 발행 스트림

# Consumer Group 설정
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "ir-group")
CONSUMER_NAME = os.getenv("CONSUMER_NAME", f"ir-{os.getpid()}")

# 경로 설정
RESULT_PATH = os.getenv("RESULT_PATH", "/results/")

# 모델 설정
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.3"))

# Provider version pins (pluggable-modules layer — see registry.py)
# Each var selects the active implementation for its slot.  The defaults
# match the current concrete implementations so existing behaviour is
# preserved.  Override via environment to swap providers without code changes.
#
# IR_MODEL_VER   : ModelProvider version selector
# IR_PROMPT_VER  : PromptProvider version selector
# IR_PARSER_VER  : Parser version selector
# IR_SCORING_VER : ScoringStrategy version selector
#
# Current defaults mirror the version constants in the existing modules:
#   plr_prompts.PROMPT_VERSION_YAML_COT = "plr_v1.3_cot"
#   query_parser.QueryJSON.parser_version default = "qp_v0.4"
#   scoring.SCORING_VERSION = "score_v0.5"
#
# IR_MODEL_VER defaults to "" (empty) = use the active model provider, whose
# version auto-derives from IR_GEMMA_REPO (see gemma_backend.gemma_model_version).
# Set it explicitly only to force a specific registered model variant.
IR_MODEL_VER   = os.getenv("IR_MODEL_VER",   "")
IR_PROMPT_VER  = os.getenv("IR_PROMPT_VER",  "plr_v1.3_cot")
IR_PARSER_VER  = os.getenv("IR_PARSER_VER",  "qp_v0.4")
IR_SCORING_VER = os.getenv("IR_SCORING_VER", "score_v0.5")

# ---------------------------------------------------------------------------
# Eager indexing guardrails
# ---------------------------------------------------------------------------

# Master on/off switch for the eager indexing consumer + reconcile daemon.
# Set IR_EAGER_ENABLED=false to disable both without code changes (e.g. during
# maintenance or on resource-constrained hosts without GPU).
IR_EAGER_ENABLED: bool = os.getenv("IR_EAGER_ENABLED", "true").lower() not in ("0", "false", "no")

# How often (seconds) the maxlen-loss reconcile daemon wakes to find and
# re-enqueue incomplete videos.  Default 300 s (5 min).
IR_RECONCILE_INTERVAL_SEC: float = float(os.getenv("IR_RECONCILE_INTERVAL_SEC", "300"))

# Max videos the reconcile daemon may re-enqueue per sweep, to avoid bursting
# the scheduler queue when many videos are stale at once.
IR_RECONCILE_MAX_PER_SWEEP: int = int(os.getenv("IR_RECONCILE_MAX_PER_SWEEP", "50"))

# Max depth of the scheduler's internal priority queue for eager indexing jobs.
# Submitting beyond this cap logs a warning and drops the job (backpressure).
# 0 = unlimited (default).
IR_EAGER_QUEUE_DEPTH_CAP: int = int(os.getenv("IR_EAGER_QUEUE_DEPTH_CAP", "0"))
