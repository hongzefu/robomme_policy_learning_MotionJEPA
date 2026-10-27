from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import TYPE_CHECKING

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from scripts.evaluation.runtime_contract import RpcSession

if TYPE_CHECKING:
    from mme_vla_suite.policies.policy import MME_VLA_Policy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: MME_VLA_Policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._eval_mode = bool(self._metadata.get("run_identity_sha256"))
        self._active_connection = None
        self._rpc_active = False
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        # 活动租约在首次await之前建立；断线保留，防止旧客户端迟到重置另一局。
        if self._eval_mode and self._active_connection is not None:
            await websocket.send(packer.pack({"error": "active_attempt"}))
            await websocket.close(code=1008, reason="已有活动attempt，须显式静止释放或重启resident")
            return
        if self._eval_mode:
            self._active_connection = websocket
        session = RpcSession(self._metadata) if self._eval_mode else None
        await websocket.send(packer.pack(self._metadata))
        
        while True:
            phase = "contract"
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                operation = session.validate(obs) if session is not None else None
                rpc = obs.get("eval_rpc")
                if operation == "release":
                    # 所有策略调用同步执行；读到释放请求说明之前的调用已完成。
                    if self._rpc_active:
                        raise ValueError("尚有执行中RPC，不能释放")
                    await websocket.send(packer.pack({"release_attempt_finished": True,
                                                      "quiescent": True, "eval_rpc": rpc}))
                    self._active_connection = None
                    await websocket.close()
                    return
                policy_obs = {key: value for key, value in obs.items() if key != "eval_rpc"}
                self._rpc_active = True
                phase = "runtime"
                
                if obs.get("reset", False):
                    tstart = time.monotonic()
                    self._policy.reset()
                    tend = time.monotonic() - tstart
                    if session is not None:
                        session.commit("reset", rpc)
                    self._rpc_active = False
                    await websocket.send(packer.pack(
                        {"reset_finished": True, "reset_time_ms": tend * 1000,
                         **({"eval_rpc": rpc} if session is not None else {})}))
                elif obs.get("add_buffer", False):
                    tstart = time.monotonic()
                    self._policy.add_buffer(policy_obs)
                    tend = time.monotonic() - tstart
                    if session is not None:
                        session.commit("add_buffer", rpc)
                    self._rpc_active = False
                    # 挂钟落日志（0901-motion-memory-plan.md 3.5 计时口径：add_buffer 含帧路编码 + device_get + 运动路同步编码）
                    logger.info("TIMING add_buffer_ms=%.1f frames=%d exec_start_idx=%s",
                                tend * 1000, len(obs.get("images", [])), obs.get("exec_start_idx"))
                    await websocket.send(packer.pack(
                        {"add_buffer_finished": True, "add_buffer_time_ms": tend * 1000,
                         **({"eval_rpc": rpc} if session is not None else {})}))
                else:
                    outputs = self._policy.infer(policy_obs)
                    if session is not None:
                        session.commit("infer", rpc)
                        outputs = {**outputs, "eval_rpc": rpc}
                    self._rpc_active = False
                    # infer_time_ms 由 policy.infer 在计时段内 block_until_ready 后给出（S3）
                    logger.info("TIMING infer_ms=%.1f", float(outputs.get("infer_time_ms", float("nan"))))
                    await websocket.send(packer.pack(outputs))

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as error:
                if self._eval_mode:
                    await websocket.send(packer.pack({"error": "contract_violation" if phase == "contract" else "runtime_failure",
                        "message": str(error), "rpc_executed": phase == "runtime"}))
                else:
                    await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
