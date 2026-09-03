"""在线 motion 编码进程（sidecar）与 policy 之间的 Unix socket 协议公共件（motion-memory-plan.md 第二部分三节）。

**只 import stdlib**：主侧 `from mme_vla_suite.policies.motion_protocol import ...`；sidecar（wan 子 venv，装不下 mme_vla_suite）用
`importlib.util.spec_from_file_location` 从同一绝对文件加载，并在握手里上报本文件 sha256——双方不得各抄常量或 `_recv_exact`。

协议 v1（统一小端）：
  握手（子 → 父）：uint32 length + JSON（provenance + protocol_sha256 + stub 标记）
  请求（父 → 子）：8 字节 magic `MMEMOT01`（直接编码版本）+ uint32 length（= 8 + payload）+ int64 起点全域帧号 + payload
                 payload = 33×256×256×3 uint8 原始字节（精确 6,488,064 B）
  响应（子 → 父）：uint32 status（0 = OK）+ 768×4 字节 f32（精确 3,072 B）；status ≠ 0 时后随 uint32 length + JSON 错误
发送统一 `sendall`；接收双方共用 `recv_exact(sock, n, deadline)`：按同一个 monotonic 总 deadline 循环 `recv_into` 直到恰好 n 字节，
禁止假设一次 `recv(n)` 能收到 6.3 MB。EOF、短包、错误 magic / 版本 / status、超长 length、整次请求超过 `REQUEST_TIMEOUT_S` 均 fail-loud。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import socket
import struct
import time

MAGIC = b"MMEMOT01"
WINDOW_FRAMES = 33
FRAME_SIZE = 256
TOKEN_DIM = 768
PAYLOAD_BYTES = WINDOW_FRAMES * FRAME_SIZE * FRAME_SIZE * 3      # 6,488,064
RESPONSE_BYTES = TOKEN_DIM * 4                                   # 3,072
MAX_LENGTH = 8 + PAYLOAD_BYTES                                   # 请求 length 上限
MAX_JSON = 1 << 20
REQUEST_TIMEOUT_S = 60.0
HANDSHAKE_TIMEOUT_S = 600.0                                      # 加载 VAE + encoder + 预热
STATUS_OK = 0
STATUS_ERROR = 1
SHUTDOWN_LENGTH = 0                                              # length == 0 的请求 = 关闭


class ProtocolError(RuntimeError):
    pass


def protocol_sha256() -> str:
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()


def recv_exact(sock: socket.socket, n: int, deadline: float) -> bytes:
    """按 monotonic 总 deadline 循环 recv_into 直到恰好 n 字节；EOF / 超时即 raise。"""
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        remain = deadline - time.monotonic()
        if remain <= 0:
            raise ProtocolError(f"recv_exact 超时：已收 {got}/{n} 字节")
        sock.settimeout(remain)
        try:
            k = sock.recv_into(mv[got:], n - got)
        except socket.timeout as e:
            raise ProtocolError(f"recv_exact 超时：已收 {got}/{n} 字节") from e
        if k == 0:
            raise ProtocolError(f"对端 EOF：已收 {got}/{n} 字节")
        got += k
    return bytes(buf)


def send_all(sock: socket.socket, data: bytes) -> None:
    sock.settimeout(REQUEST_TIMEOUT_S)
    sock.sendall(data)


# ── 握手 ──────────────────────────────────────────────────────────────────────

def send_handshake(sock: socket.socket, payload: dict) -> None:
    raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    send_all(sock, struct.pack("<I", len(raw)) + raw)


def recv_handshake(sock: socket.socket, deadline: float) -> dict:
    (n,) = struct.unpack("<I", recv_exact(sock, 4, deadline))
    if n == 0 or n > MAX_JSON:
        raise ProtocolError(f"握手 length 非法: {n}")
    return json.loads(recv_exact(sock, n, deadline).decode("utf-8"))


# ── 请求 / 响应 ───────────────────────────────────────────────────────────────

def encode_request(start_frame: int, payload: bytes) -> bytes:
    if len(payload) != PAYLOAD_BYTES:
        raise ProtocolError(f"payload {len(payload)} B != {PAYLOAD_BYTES}")
    return MAGIC + struct.pack("<I", 8 + PAYLOAD_BYTES) + struct.pack("<q", int(start_frame)) + payload


def encode_shutdown() -> bytes:
    return MAGIC + struct.pack("<I", SHUTDOWN_LENGTH)


def recv_request(sock: socket.socket, deadline: float) -> tuple[int, bytes] | None:
    """返回 (start_frame, payload)；关闭请求返回 None。"""
    head = recv_exact(sock, 12, deadline)
    if head[:8] != MAGIC:
        raise ProtocolError(f"错误 magic / 版本: {head[:8]!r}")
    (n,) = struct.unpack("<I", head[8:12])
    if n == SHUTDOWN_LENGTH:
        return None
    if n != MAX_LENGTH:
        raise ProtocolError(f"请求 length {n} != {MAX_LENGTH}")
    (start,) = struct.unpack("<q", recv_exact(sock, 8, deadline))
    payload = recv_exact(sock, PAYLOAD_BYTES, deadline)
    return int(start), payload


def encode_response(token_bytes: bytes) -> bytes:
    if len(token_bytes) != RESPONSE_BYTES:
        raise ProtocolError(f"token {len(token_bytes)} B != {RESPONSE_BYTES}")
    return struct.pack("<I", STATUS_OK) + token_bytes


def encode_error(msg: str) -> bytes:
    raw = msg.encode("utf-8")[:MAX_JSON]
    return struct.pack("<I", STATUS_ERROR) + struct.pack("<I", len(raw)) + raw


def recv_response(sock: socket.socket, deadline: float) -> bytes:
    (status,) = struct.unpack("<I", recv_exact(sock, 4, deadline))
    if status == STATUS_OK:
        return recv_exact(sock, RESPONSE_BYTES, deadline)
    if status == STATUS_ERROR:
        (n,) = struct.unpack("<I", recv_exact(sock, 4, deadline))
        if n > MAX_JSON:
            raise ProtocolError(f"错误消息 length 非法: {n}")
        raise ProtocolError("sidecar 报错: " + recv_exact(sock, n, deadline).decode("utf-8", "replace"))
    raise ProtocolError(f"未知 status {status}")


# ── stub 帧编码（P1–P4：合成帧把全域帧号写进像素，通道 0 低 8 位、通道 1 高位）────────

def stub_frame(frame_idx: int) -> "np.ndarray":  # noqa: F821
    import numpy as np
    f = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), np.uint8)
    f[:, :, 0] = frame_idx & 0xFF
    f[:, :, 1] = (frame_idx >> 8) & 0xFF
    f[:, :, 2] = (frame_idx >> 16) & 0xFF
    return f


def stub_decode(window: "np.ndarray") -> list[int]:  # noqa: F821
    import numpy as np
    w = np.asarray(window)
    return [int(w[i, 0, 0, 0]) | (int(w[i, 0, 0, 1]) << 8) | (int(w[i, 0, 0, 2]) << 16) for i in range(w.shape[0])]
