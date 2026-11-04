"""
====================================================================
Student Performance Assessment System
====================================================================

Script:
    scripts/smoke_test.py

Purpose:
    Verify a RUNNING deployment, not just an importable codebase.

    After `docker compose up -d` (or any other launch), this answers:
    is the process serving, is the health endpoint healthy, does the
    HTML shell look like the app, and how long did it take?

Usage
-----
    python scripts/smoke_test.py
    python scripts/smoke_test.py --url https://spas.university.edu.gh

Exit codes
----------
    0  all checks passed
    1  a check failed (deployment is not serving correctly)
====================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request


def fetch(url: str, timeout: float = 10.0) -> tuple[int, bytes]:

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "spas-smoke-test/1.0"},
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:

        return response.status, response.read()


def main(argv=None) -> int:

    parser = argparse.ArgumentParser(
        description="Smoke-test a running deployment."
    )

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8501",
        help="base URL of the deployment (default: %(default)s)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="per-request timeout in seconds",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable output (for CI)",
    )

    args = parser.parse_args(argv)

    base = args.url.rstrip("/")

    checks = []

    def check(name: str, fn) -> None:

        started = time.perf_counter()

        try:

            detail = fn()

            checks.append(
                {
                    "check": name,
                    "status": "pass",
                    "detail": detail,
                    "ms": round(
                        (time.perf_counter() - started) * 1000, 1
                    ),
                }
            )

        except Exception as error:

            checks.append(
                {
                    "check": name,
                    "status": "fail",
                    "detail": str(error),
                    "ms": round(
                        (time.perf_counter() - started) * 1000, 1
                    ),
                }
            )

    # ---- 1. health endpoint ---------------------------------
    def health():

        status, body = fetch(
            f"{base}/_stcore/health",
            args.timeout,
        )

        body = body.strip()

        if status != 200 or body != b"ok":
            raise RuntimeError(
                f"unexpected health response: {status} {body!r}"
            )

        return "ok"

    # ---- 2. index page serves HTML ---------------------------
    def index():

        status, body = fetch(f"{base}/", args.timeout)

        if status != 200:
            raise RuntimeError(f"HTTP {status} from /")

        text = body.decode("utf-8", errors="ignore")

        for marker in ("streamlit", "<html"):
            if marker not in text.lower():
                raise RuntimeError(
                    f"/ does not look like the app (missing {marker!r})"
                )

        return f"{len(body)} bytes"

    # ---- 3. front-end bundle reachable -----------------------
    def static():
        """Find the hashed JS bundle in index.html and fetch it.

        Streamlit serves a catch-all HTML shell for unknown paths, so a
        naive "/static/js/index.js" probe can return 200 with HTML and
        look like a pass. This checks the real, hashed asset instead.
        """

        _, html = fetch(f"{base}/", args.timeout)
        text = html.decode("utf-8", errors="ignore")

        match = re.search(r"/static/js/index\.[A-Za-z0-9_-]+\.js", text)

        if not match:
            raise RuntimeError(
                "no hashed JS bundle referenced in index.html — the "
                "front-end build is missing from the image"
            )

        asset = match.group(0)
        status, body = fetch(f"{base}{asset}", args.timeout)

        if status != 200:
            raise RuntimeError(f"HTTP {status} from {asset}")

        head = body[:200].lstrip().lower()

        if head.startswith(b"<!doctype") or head.startswith(b"<html"):
            raise RuntimeError(
                f"{asset} returned HTML, not JavaScript (static serving "
                "is broken or the file is missing)"
            )

        return f"{asset} ({len(body) // 1024} KB of JavaScript)"

    # ---- 4. XSRF/CORS posture -------------------------------
    def headers():

        request = urllib.request.Request(
            f"{base}/_stcore/health",
            headers={"User-Agent": "spas-smoke-test/1.0"},
        )

        with urllib.request.urlopen(
            request, timeout=args.timeout
        ) as response:

            return f"server={response.headers.get('server', '?')}"

    check("health endpoint returns ok", health)
    check("index page serves the app shell", index)
    check("static bundle is reachable", static)
    check("response headers readable", headers)

    failed = [item for item in checks if item["status"] == "fail"]

    if args.json:

        print(json.dumps(checks, indent=2))

    else:

        print("=" * 60)
        print(f"SMOKE TEST — {base}")
        print("=" * 60)

        for item in checks:

            icon = "✅" if item["status"] == "pass" else "❌"

            print(
                f"{icon} {item['check']} "
                f"({item['ms']} ms) — {item['detail']}"
            )

        print("-" * 60)

        if failed:
            print(f"FAILED — {len(failed)} check(s) did not pass.")
        else:
            print("PASSED — the deployment is serving correctly.")

    return 1 if failed else 0


if __name__ == "__main__":

    raise SystemExit(main())
