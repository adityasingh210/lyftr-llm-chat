"""Browser end-to-end check of the deployed (or local) UI against the frontend contract.

Usage:
    pip install playwright && playwright install chromium
    python scripts/e2e_ui.py http://localhost:3000 [--screenshots out_dir]

Requires the backend to have ENABLE_FAILURE_SIMULATION=true for the failover steps.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

TIMEOUT = 60_000


def main(url: str, shots: Path | None) -> None:
    question = "What is the capital of France?"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 860})
        page.set_default_timeout(TIMEOUT)

        def shot(name: str) -> None:
            if shots:
                shots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(shots / f"{name}.png"))

        # 1) live inference, input cleared, metadata rendered
        page.goto(url)
        expect(page.get_by_test_id("health-status")).to_contain_text("keys")
        page.get_by_test_id("chat-input").fill(question)
        page.get_by_test_id("send-button").click()
        expect(page.get_by_test_id("chat-input")).to_have_value("")
        first = page.get_by_test_id("assistant-message").nth(0)
        expect(first).to_be_visible()
        source1 = first.get_by_test_id("meta-source").inner_text()
        for tid in ["meta-model", "meta-latency", "meta-input-tokens", "meta-output-tokens"]:
            expect(first.get_by_test_id(tid)).not_to_be_empty()
        expect(first.get_by_test_id("meta-latency")).to_have_text(re.compile(r"^\d+ ms$"))
        expect(first.get_by_test_id("meta-input-tokens")).to_have_text(re.compile(r"^\d+$"))
        print(f"[ok] first reply: source={source1}")

        # 2) multi-turn in the same session (Enter sends)
        page.get_by_test_id("chat-input").fill("And what did I just ask you?")
        page.get_by_test_id("chat-input").press("Enter")
        expect(page.get_by_test_id("assistant-message")).to_have_count(2)
        print("[ok] second turn via Enter")
        shot("01-multi-turn")

        # 3) reload = new session; same first message is now a cache hit
        page.reload()
        expect(page.get_by_test_id("assistant-message")).to_have_count(0)
        page.get_by_test_id("chat-input").fill(question)
        page.get_by_test_id("send-button").click()
        cached = page.get_by_test_id("assistant-message").nth(0)
        expect(cached.get_by_test_id("meta-source")).to_have_text("LLM_CACHE")
        print(f"[ok] cache hit on new session, latency={cached.get_by_test_id('meta-latency').inner_text()}")
        shot("02-cache-hit")

        # 4) failover via the X-Simulate-Failure control
        page.get_by_role("button", name="Settings").click()
        page.get_by_test_id("simulate-failure").select_option("key_1=429")
        page.get_by_test_id("chat-input").fill("Failover test: say hi")
        page.get_by_test_id("send-button").click()
        failover = page.get_by_test_id("assistant-message").nth(1)
        expect(failover.get_by_test_id("meta-key-id")).to_have_text("key_2")
        expect(failover.get_by_test_id("meta-source")).to_have_text("LIVE_INFERENCE")
        print("[ok] key_1=429 answered by key_2")
        shot("03-failover")

        # 5) every key down -> readable 503 in the chat
        page.get_by_test_id("simulate-failure").select_option("*=429")
        page.get_by_test_id("chat-input").fill("This one should fail")
        page.get_by_test_id("send-button").click()
        expect(page.get_by_test_id("error-message")).to_contain_text("unavailable")
        print("[ok] 503 rendered as a chat message")
        shot("04-all-keys-down")

        # 6) system prompt field exists and is sent
        page.get_by_test_id("simulate-failure").select_option("")
        page.get_by_test_id("system-prompt-input").fill("You answer in exactly one short sentence.")
        page.get_by_test_id("chat-input").fill("Describe the ocean.")
        page.get_by_test_id("send-button").click()
        expect(page.get_by_test_id("assistant-message")).to_have_count(3)
        print("[ok] system prompt turn")
        shot("05-system-prompt")

        browser.close()
    print("UI e2e: all checks passed")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
    out = Path(sys.argv[sys.argv.index("--screenshots") + 1]) if "--screenshots" in sys.argv else None
    main(target, out)
