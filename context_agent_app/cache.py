"""
Caching layer for the multi-agent system.
Supports both in-memory and Redis backends.
"""
import hashlib
import json
import time
from typing import Any, Optional, Dict
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from cachetools import TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    logger.warning("cachetools not installed. In-memory caching will use basic dict.")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not installed. Redis caching unavailable.")


class CacheStats:
    """Track cache statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.errors = 0
        
    def record_hit(self):
        self.hits += 1
        
    def record_miss(self):
        self.misses += 1
        
    def record_set(self):
        self.sets += 1
        
    def record_error(self):
        self.errors += 1
        
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "total_requests": self.hits + self.misses
        }


class BaseCache:
    """Base cache interface."""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.stats = CacheStats()
        
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        raise NotImplementedError
        
    def delete(self, key: str) -> bool:
        raise NotImplementedError
        
    def clear(self) -> bool:
        raise NotImplementedError
        
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.to_dict()


class InMemoryCache(BaseCache):
    """In-memory cache using TTLCache or basic dict."""
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        super().__init__(ttl)
        if CACHETOOLS_AVAILABLE:
            self._cache = TTLCache(maxsize=max_size, ttl=ttl)
        else:
            # Fallback to basic dict with manual TTL tracking
            self._cache = {}
            self._expiry = {}
        self.max_size = max_size
        
    def get(self, key: str) -> Optional[Any]:
        try:
            if CACHETOOLS_AVAILABLE:
                value = self._cache.get(key)
            else:
                # Check expiry for basic dict
                if key in self._expiry and time.time() > self._expiry[key]:
                    del self._cache[key]
                    del self._expiry[key]
                    value = None
                else:
                    value = self._cache.get(key)
                    
            if value is not None:
                self.stats.record_hit()
                logger.debug(f"Cache hit for key: {key[:50]}...")
            else:
                self.stats.record_miss()
                logger.debug(f"Cache miss for key: {key[:50]}...")
            return value
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Cache get error: {e}")
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            if CACHETOOLS_AVAILABLE:
                self._cache[key] = value
            else:
                # Manual TTL tracking for basic dict
                self._cache[key] = value
                self._expiry[key] = time.time() + (ttl or self.ttl)
                
                # Simple size limit enforcement
                if len(self._cache) > self.max_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    if oldest_key in self._expiry:
                        del self._expiry[oldest_key]
                        
            self.stats.record_set()
            logger.debug(f"Cache set for key: {key[:50]}...")
            return True
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Cache set error: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        try:
            if key in self._cache:
                del self._cache[key]
                if not CACHETOOLS_AVAILABLE and key in self._expiry:
                    del self._expiry[key]
                return True
            return False
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Cache delete error: {e}")
            return False
            
    def clear(self) -> bool:
        try:
            self._cache.clear()
            if not CACHETOOLS_AVAILABLE:
                self._expiry.clear()
            return True
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Cache clear error: {e}")
            return False


class RedisCache(BaseCache):
    """Redis-backed cache."""
    
    def __init__(self, ttl: int = 3600, redis_url: str = "redis://localhost:6379/0"):
        super().__init__(ttl)
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Install with: pip install redis")
        self.client = redis.from_url(redis_url, decode_responses=True)
        
    def get(self, key: str) -> Optional[Any]:
        try:
            value_str = self.client.get(key)
            if value_str:
                self.stats.record_hit()
                logger.debug(f"Cache hit for key: {key[:50]}...")
                return json.loads(value_str)
            else:
                self.stats.record_miss()
                logger.debug(f"Cache miss for key: {key[:50]}...")
                return None
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Redis get error: {e}")
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            value_str = json.dumps(value)
            self.client.setex(key, ttl or self.ttl, value_str)
            self.stats.record_set()
            logger.debug(f"Cache set for key: {key[:50]}...")
            return True
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Redis set error: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Redis delete error: {e}")
            return False
            
    def clear(self) -> bool:
        try:
            self.client.flushdb()
            return True
        except Exception as e:
            self.stats.record_error()
            logger.error(f"Redis clear error: {e}")
            return False


class CacheManager:
    """Manages multiple cache instances for different purposes."""
    
    def __init__(self, backend: str = "memory", redis_url: str = "redis://localhost:6379/0"):
        self.backend = backend
        self.redis_url = redis_url
        self.caches: Dict[str, BaseCache] = {}
        
    def get_cache(self, name: str, ttl: int = 3600, max_size: int = 1000) -> BaseCache:
        """Get or create a named cache instance."""
        if name not in self.caches:
            if self.backend == "redis":
                self.caches[name] = RedisCache(ttl=ttl, redis_url=self.redis_url)
            else:
                self.caches[name] = InMemoryCache(ttl=ttl, max_size=max_size)
        return self.caches[name]
        
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {name: cache.get_stats() for name, cache in self.caches.items()}
        
    def clear_all(self) -> bool:
        """Clear all caches."""
        success = True
        for cache in self.caches.values():
            success = success and cache.clear()
        return success


# Utility functions for cache key generation
def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a cache key from arguments.
    
    Args:
        prefix: Key prefix (e.g., "entity", "web_fetch")
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key
        
    Returns:
        Cache key string
    """
    # Create a stable string representation
    key_parts = [prefix]
    
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            # Hash complex objects
            key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest())
            
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(f"{k}={hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest()}")
            
    return ":".join(key_parts)


def hash_text(text: str) -> str:
    """Generate a hash for text content."""
    return hashlib.sha256(text.encode()).hexdigest()


# Decorator for caching function results
def cached(cache_name: str, ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator to cache function results.
    
    Args:
        cache_name: Name of the cache to use
        ttl: Optional TTL override
        key_prefix: Prefix for cache keys
        
    Example:
        @cached("entity_cache", ttl=3600, key_prefix="extract")
        async def extract_entities(text):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get cache manager from config
            from context_agent_app.config import CACHE_ENABLED
            if not CACHE_ENABLED:
                return await func(*args, **kwargs)
                
            # Generate cache key
            cache_key = generate_cache_key(key_prefix or func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cache = _get_cache_instance(cache_name, ttl)
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
                
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            from context_agent_app.config import CACHE_ENABLED
            if not CACHE_ENABLED:
                return func(*args, **kwargs)
                
            cache_key = generate_cache_key(key_prefix or func.__name__, *args, **kwargs)
            cache = _get_cache_instance(cache_name, ttl)
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
                
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result
            
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        from context_agent_app.config import CACHE_BACKEND, REDIS_URL
        _cache_manager = CacheManager(backend=CACHE_BACKEND, redis_url=REDIS_URL)
    return _cache_manager


def _get_cache_instance(name: str, ttl: Optional[int] = None) -> BaseCache:
    """Internal helper to get cache instance."""
    from context_agent_app.config import MAX_CACHE_SIZE
    manager = get_cache_manager()
    return manager.get_cache(name, ttl=ttl or 3600, max_size=MAX_CACHE_SIZE)
