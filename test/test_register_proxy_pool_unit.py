import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

curl_cffi_module = types.ModuleType("curl_cffi")
curl_cffi_module.requests = types.SimpleNamespace()
sys.modules.setdefault("curl_cffi", curl_cffi_module)

config_module = types.ModuleType("services.config")
config_module.DATA_DIR = Path("C:/tmp/chatgpt2api-register-unit-test")
sys.modules.setdefault("services.config", config_module)

proxy_service_module = types.ModuleType("services.proxy_service")


def _normalize_proxy_url(url: str) -> str:
    candidate = str(url or "").strip()
    if candidate and "://" not in candidate:
        parts = candidate.split(":", 3)
        if len(parts) == 2 and parts[1].isdigit():
            candidate = f"http://{candidate}"
    if candidate.lower().startswith("socks5://"):
        return "socks5h://" + candidate[len("socks5://") :]
    return candidate


proxy_service_module.normalize_proxy_url = _normalize_proxy_url
sys.modules.setdefault("services.proxy_service", proxy_service_module)

account_service_module = types.ModuleType("services.account_service")
account_service_module.account_service = types.SimpleNamespace(list_accounts=lambda: [])
sys.modules.setdefault("services.account_service", account_service_module)

openai_register_module = types.ModuleType("services.register.openai_register")
openai_register_module.config = {
    "mail": {
        "request_timeout": 30,
        "wait_timeout": 30,
        "wait_interval": 2,
        "api_use_register_proxy": True,
        "providers": [],
    },
    "proxy": "",
    "proxy_input_mode": "single",
    "proxy_url": "",
    "proxy_list_text": "",
    "proxy_refresh_interval": 120,
    "total": 10,
    "threads": 3,
}
openai_register_module.stats = {
    "current_proxy": "",
    "proxy_pool_count": 0,
    "proxy_source": "single",
    "proxy_pool_last_error": "",
    "proxy_pool_last_fetch": 0,
}
openai_register_module.stats_lock = threading.RLock()
openai_register_module.register_log_sink = None
openai_register_module.configure_proxy_pool = lambda fetch_now=False: dict(openai_register_module.stats)
openai_register_module.prepare_proxy_pool = lambda: dict(openai_register_module.stats)
openai_register_module.worker = lambda index: {"ok": False}
sys.modules.setdefault("services.register.openai_register", openai_register_module)

from services.register.proxy_pool import RegisterProxyPool, normalize_proxy_input_mode, parse_proxy_lines
from services.register.mail_provider import CloudMailGenProvider, cloudmail_token_cache
from services.register_service import _normalize


class RegisterProxyPoolUnitTests(unittest.TestCase):
    def test_normalizes_input_mode(self) -> None:
        self.assertEqual(normalize_proxy_input_mode(" URL "), "url")
        self.assertEqual(normalize_proxy_input_mode("TEXT"), "text")
        self.assertEqual(normalize_proxy_input_mode("bad"), "single")

    def test_parse_proxy_lines_normalizes_and_deduplicates(self) -> None:
        self.assertEqual(
            parse_proxy_lines("127.0.0.1:7890\n127.0.0.1:7890, socks5://proxy.example:1080"),
            ["http://127.0.0.1:7890", "socks5h://proxy.example:1080"],
        )

    def test_prepare_rejects_empty_text_pool(self) -> None:
        pool = RegisterProxyPool()
        pool.configure(mode="text", single_proxy="", proxy_url="", proxy_list_text="", refresh_interval=120)

        with self.assertRaisesRegex(RuntimeError, "no proxies available"):
            pool.prepare()

        self.assertIn("no proxies available", pool.state().last_error)

    def test_prepare_rejects_url_fetch_error_even_with_existing_proxies(self) -> None:
        pool = RegisterProxyPool()
        pool.configure(
            mode="url",
            single_proxy="",
            proxy_url="https://example.com/proxies.txt",
            proxy_list_text="",
            refresh_interval=120,
        )
        with patch.object(pool, "refresh_url", side_effect=lambda force: pool._record_error("failed to fetch proxy URL: boom")):
            with self.assertRaisesRegex(RuntimeError, "failed to fetch proxy URL"):
                pool.prepare()


class RegisterServiceNormalizeUnitTests(unittest.TestCase):
    def test_infers_url_mode_from_legacy_proxy_url(self) -> None:
        normalized = _normalize({"proxy_url": " https://example.com/proxies.txt "})

        self.assertEqual(normalized["proxy_input_mode"], "url")
        self.assertEqual(normalized["proxy_url"], "https://example.com/proxies.txt")

    def test_migrates_legacy_multiline_proxy_to_text_mode(self) -> None:
        normalized = _normalize({"proxy": "127.0.0.1:7890\nhttp://proxy.example:8080"})

        self.assertEqual(normalized["proxy_input_mode"], "text")
        self.assertEqual(normalized["proxy_list_text"], "127.0.0.1:7890\nhttp://proxy.example:8080")
        self.assertEqual(normalized["proxy"], "")

    def test_defaults_mail_api_proxy_toggle_to_true_and_preserves_false(self) -> None:
        defaulted = _normalize({"mail": {"providers": []}})
        disabled = _normalize({"mail": {"api_use_register_proxy": False, "providers": []}})

        self.assertIs(defaulted["mail"]["api_use_register_proxy"], True)
        self.assertIs(disabled["mail"]["api_use_register_proxy"], False)


class CloudMailGenProviderUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        cloudmail_token_cache.clear()

    def _provider(self) -> CloudMailGenProvider:
        provider = CloudMailGenProvider.__new__(CloudMailGenProvider)
        provider.api_base = "https://mail.example"
        provider.admin_email = "admin@example.com"
        provider.admin_password = "secret"
        provider.conf = {"user_agent": "test-agent", "request_timeout": 1}
        return provider

    def test_email_list_business_error_refreshes_cached_token_once(self) -> None:
        provider = self._provider()
        cache_key = provider._cache_key()
        cloudmail_token_cache[cache_key] = ("stale-token", time.time() + 3600)
        email_list_tokens: list[str] = []
        gen_token_calls = 0

        def fake_request(method, path, headers=None, params=None, payload=None, expected=(200,)):
            nonlocal gen_token_calls
            if path == "/api/public/genToken":
                gen_token_calls += 1
                return {"code": 200, "data": {"token": "fresh-token"}}
            if path == "/api/public/emailList":
                email_list_tokens.append(str((headers or {}).get("Authorization") or ""))
                if len(email_list_tokens) == 1:
                    return {"code": 401, "message": "token expired"}
                return {
                    "code": 200,
                    "data": [
                        {
                            "emailId": "m-1",
                            "toEmail": "user@example.com",
                            "subject": "Verification code",
                            "text": "code is 123456",
                            "sendEmail": "noreply@example.com",
                            "createTime": "2026-06-15T00:00:00Z",
                        }
                    ],
                }
            raise AssertionError(f"unexpected path: {path}")

        provider._request = fake_request

        message = provider.fetch_latest_message({"address": "user@example.com"})

        self.assertEqual(email_list_tokens, ["stale-token", "fresh-token"])
        self.assertEqual(gen_token_calls, 1)
        self.assertIsNotNone(message)
        self.assertEqual(message["message_id"], "m-1")
        self.assertEqual(message["sender"], "noreply@example.com")

    def test_request_retries_retryable_status(self) -> None:
        provider = self._provider()

        class FakeResponse:
            def __init__(self, status_code: int, payload: dict, text: str = ""):
                self.status_code = status_code
                self._payload = payload
                self.text = text

            def json(self):
                return self._payload

        class FakeSession:
            def __init__(self):
                self.calls = 0

            def request(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return FakeResponse(500, {}, "temporary failure")
                return FakeResponse(200, {"ok": True})

        fake_session = FakeSession()
        provider.session = fake_session

        with patch("services.register.mail_provider.time.sleep"):
            data = provider._request("GET", "/api/public/ping")

        self.assertEqual(data, {"ok": True})
        self.assertEqual(fake_session.calls, 2)


if __name__ == "__main__":
    unittest.main()
