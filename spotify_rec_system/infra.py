import json
import os
from typing import Any, Dict, List, Optional

# 可选依赖：Kafka 与 Redis 均为按需启用，未配置时自动降级为 no-op。
try:
    from kafka import KafkaProducer  # type: ignore
except Exception:  # pragma: no cover - 在缺省环境下可能不存在
    KafkaProducer = None

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None


class EventProducer:
    """Kafka 事件生产者，用于近线层行为上报。"""

    def __init__(self, topic: Optional[str] = None, bootstrap_servers: Optional[str] = None):
        self.topic = topic or os.getenv("KAFKA_TOPIC_EVENTS", "spotify_events")
        brokers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS")

        if KafkaProducer is None or not brokers:
            self.enabled = False
            self.producer = None
            return

        try:
            self.producer = KafkaProducer(
                bootstrap_servers=brokers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                linger_ms=10,
            )
            self.enabled = True
        except Exception as exc:  # pragma: no cover - 连接失败时降级
            print(f"[WARN] Kafka 初始化失败，关闭事件上报: {exc}")
            self.enabled = False
            self.producer = None

    def send_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        if not self.enabled or not self.producer:
            return False
        event = {"type": event_type, **payload}
        try:
            self.producer.send(self.topic, event)
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] 发送 Kafka 事件失败: {exc}")
            return False


class RedisFeatureStore:
    """Redis 封装，用于实时/在线特征或推荐缓存。"""

    def __init__(self, url: Optional[str] = None, namespace: str = "rec"):  # noqa: D401
        redis_url = url or os.getenv("REDIS_URL")
        self.namespace = namespace

        if redis is None or not redis_url:
            self.enabled = False
            self.client = None
            return

        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            self.enabled = True
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Redis 初始化失败，关闭缓存: {exc}")
            self.enabled = False
            self.client = None

    def _key(self, *parts: str) -> str:
        return ":".join([self.namespace, *parts])

    def cache_recommendation(self, user_id: str, playlist_id: str, track_ids: List[str], ttl_seconds: int = 900):
        if not self.enabled or not self.client:
            return False
        key = self._key("rec", user_id, playlist_id)
        try:
            self.client.set(key, json.dumps(track_ids), ex=ttl_seconds)
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Redis 缓存失败: {exc}")
            return False

    def get_cached_recommendation(self, user_id: str, playlist_id: str) -> Optional[list[str]]:
        if not self.enabled or not self.client:
            return None
        key = self._key("rec", user_id, playlist_id)
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception:  # pragma: no cover
            return None

    def store_user_features(self, user_id: str, feature_vector: List[float], ttl_seconds: int = 3600):
        if not self.enabled or not self.client:
            return False
        key = self._key("uf", user_id)
        try:
            self.client.set(key, json.dumps(feature_vector), ex=ttl_seconds)
            return True
        except Exception:  # pragma: no cover
            return False

    def get_user_features(self, user_id: str) -> Optional[List[float]]:
        if not self.enabled or not self.client:
            return None
        key = self._key("uf", user_id)
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception:  # pragma: no cover
            return None
