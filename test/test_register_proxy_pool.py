import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import register_service as register_service_module
from services.register import openai_register
from services.register.proxy_pool import RegisterProxyPool
from services.register_service import RegisterService, _normalize


class FakeThread:
    instances: list["FakeThread"] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.alive = False
        FakeThread.instances.append(self)

    def start(self):
        self.started = True
        self.alive = True

    def is_alive(self):
        return self.alive


class RegisterProxyPoolIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_config = dict(openai_register.config)
        self._original_stats = dict(openai_register.stats)
        self._original_proxy_pool = openai_register.proxy_pool
        openai_register.proxy_pool = RegisterProxyPool()
        FakeThread.instances = []

    def tearDown(self) -> None:
        openai_register.config.clear()
        openai_register.config.update(self._original_config)
        openai_register.stats.clear()
        openai_register.stats.update(self._original_stats)
        openai_register.proxy_pool = self._original_proxy_pool

    def _make_service(self, config: dict) -> tuple[tempfile.TemporaryDirectory[str], RegisterService]:
        tmp_dir = tempfile.TemporaryDirectory()
        path = Path(tmp_dir.name) / "register.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return tmp_dir, RegisterService(path)

    def test_normalize_lowercases_proxy_input_mode(self) -> None:
        normalized = _normalize({"proxy_input_mode": " URL "})

        self.assertEqual(normalized["proxy_input_mode"], "url")

    def test_start_rejects_empty_url_proxy_pool_without_starting_runner(self) -> None:
        tmp_dir, service = self._make_service({"proxy_input_mode": "url", "proxy_url": "", "total": 1, "threads": 1})
        with tmp_dir, patch.object(register_service_module.threading, "Thread", FakeThread):
            with self.assertRaisesRegex(RuntimeError, "proxy_url|no proxies"):
                service.start()

            snapshot = service.get()
            self.assertFalse(snapshot["enabled"])
            self.assertEqual(snapshot["stats"]["running"], 0)
            self.assertEqual(len(FakeThread.instances), 0)

    def test_start_returns_proxy_pool_stats_for_sse_consumers(self) -> None:
        tmp_dir, service = self._make_service(
            {
                "proxy_input_mode": "text",
                "proxy_list_text": "127.0.0.1:7890\nhttp://user:pass@example.com:8080",
                "total": 1,
                "threads": 1,
            }
        )
        with tmp_dir, patch.object(register_service_module.threading, "Thread", FakeThread):
            snapshot = service.start()

        self.assertTrue(snapshot["enabled"])
        self.assertEqual(snapshot["stats"]["proxy_pool_count"], 2)
        self.assertEqual(snapshot["stats"]["proxy_source"], "text")
        self.assertEqual(snapshot["stats"]["proxy_pool_last_error"], "")

    def test_worker_no_proxy_return_updates_global_stats(self) -> None:
        openai_register.config.update({"proxy_input_mode": "text", "proxy_list_text": ""})
        openai_register.configure_proxy_pool(fetch_now=False)
        with openai_register.stats_lock:
            openai_register.stats.update({"done": 0, "success": 0, "fail": 0, "start_time": 1.0})

        result = openai_register.worker(1)

        self.assertFalse(result["ok"])
        self.assertEqual(openai_register.stats["done"], 1)
        self.assertEqual(openai_register.stats["fail"], 1)


if __name__ == "__main__":
    unittest.main()
