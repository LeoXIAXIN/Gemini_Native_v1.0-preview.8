#!/usr/bin/env python3
"""Exercise redis-py against Gemini Native's bounded RESP2 state store."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import redis


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.chingmu_redis import ChingMuRedisMocapBuffer  # noqa: E402
from src.adapters.state_store import RespServer  # noqa: E402


def main() -> int:
    server = RespServer(("127.0.0.1", 0))
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        generated = time.monotonic()
        qpos = np.zeros(36, dtype=np.float32)
        qpos[2] = 0.8
        qpos[3] = 1.0
        packet = {
            "schema_version": 1,
            "generated_monotonic": generated,
            "qpos": qpos.tolist(),
        }
        writer = redis.Redis(
            host=host,
            port=port,
            db=0,
            decode_responses=False,
            socket_timeout=1.0,
            protocol=2,
        )
        assert writer.ping() is True
        assert writer.set("action_qpos_g1_packet", json.dumps(packet)) is True

        # Do not inject a fake Redis object: this constructor is the regression
        # boundary that previously sent HELLO and crashed the real controller.
        buffer = ChingMuRedisMocapBuffer(host=host, port=port)
        observed, observed_at = buffer.read()
        assert np.array_equal(observed, qpos)
        assert observed_at == generated
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    print("LOCAL RESP2 / REDIS-PY COMPATIBILITY TEST PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
