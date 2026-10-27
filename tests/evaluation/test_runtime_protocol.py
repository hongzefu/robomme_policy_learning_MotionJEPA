"""RPC前置校验与真实本机websocket占用/断线故障测试。"""

import asyncio

import numpy as np
from openpi_client import msgpack_numpy
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from mme_vla_suite.serving.websocket_policy_server import WebsocketPolicyServer
from scripts.evaluation.runtime_contract import RpcSession, payload_sha256

META = {"run_identity_sha256": "a" * 64, "server_nonce": "server", "model_seed": 42, "model_kind": "stage2_pool4_concat"}


def reset(nonce="episode"):
    return {"reset": True, "eval_rpc": {"episode_nonce": nonce,
        "expected_run_identity_sha256": META["run_identity_sha256"], "expected_server_nonce": "server"}}


def batch():
    obs = {"add_buffer": True, "images": np.zeros((1, 2, 2, 3), np.uint8), "state": np.zeros((1, 8), np.float32), "exec_start_idx": 0}
    obs["eval_rpc"] = {"episode_nonce": "episode", "batch_seq": 0, "first_global_frame": 0,
                       "frame_count": 1, "payload_sha256": payload_sha256(obs)}
    return obs


def infer():
    return {"observation/state": np.zeros(8, np.float32),
            "eval_rpc": {"episode_nonce": "episode", "replan_seq": 0, "last_global_frame": 0}}


def test_hash_stable_through_wire_and_detects_type_or_pixels():
    obs = batch()
    unpacked = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(obs))
    assert payload_sha256(obs) == payload_sha256(unpacked)
    unpacked["state"] = unpacked["state"].astype(np.float64)
    assert payload_sha256(obs) != payload_sha256(unpacked)


@pytest.mark.parametrize("mutation", ["run", "server", "nonce", "batch_seq", "first_global_frame", "frame_count", "payload_sha256", "partial_demo"])
def test_contract_rejects_before_state_commit(mutation):
    session = RpcSession(META)
    obs = reset()
    if mutation in ("run", "server"):
        obs["eval_rpc"]["expected_run_identity_sha256" if mutation == "run" else "expected_server_nonce"] = "wrong"
    else:
        session.commit(session.validate(obs), obs["eval_rpc"])
        obs = batch()
        if mutation == "nonce":
            obs["eval_rpc"]["episode_nonce"] = "old"
        elif mutation == "payload_sha256":
            obs["images"][0, 0, 0, 0] = 1
        elif mutation == "partial_demo":
            obs["exec_start_idx"] = 1
            obs["eval_rpc"]["payload_sha256"] = payload_sha256(obs)
        else:
            obs["eval_rpc"][mutation] = 9
    with pytest.raises(ValueError):
        session.validate(obs)
    assert session.next_frame == session.replan_seq == 0


def test_infer_cannot_replay_rng_or_use_future_frame():
    session = RpcSession(META)
    for obs in (reset(), batch()):
        session.commit(session.validate(obs), obs["eval_rpc"])
    obs = infer()
    session.commit(session.validate(obs), obs["eval_rpc"])
    with pytest.raises(ValueError):
        session.validate(obs)
    obs["eval_rpc"]["replan_seq"] = 1
    with pytest.raises(ValueError):
        session.validate(obs)


class FakePolicy:
    def __init__(self):
        self.calls = []
        self.done = None
    def record(self, operation):
        self.calls.append(operation)
        if self.done is not None:
            self.done.set()
    def reset(self):
        self.record("reset")
    def add_buffer(self, obs):
        assert "eval_rpc" not in obs
        self.record("add_buffer")
    def infer(self, obs):
        assert "eval_rpc" not in obs
        self.record("infer")
        return {"actions": np.zeros((20, 8), np.float32)}


async def rpc(socket, obs):
    await socket.send(msgpack_numpy.Packer().pack(obs))
    return msgpack_numpy.unpackb(await socket.recv())


def test_real_socket_single_connection_release_and_reconnect():
    async def scenario():
        policy = FakePolicy()
        server = WebsocketPolicyServer(policy, metadata=META)
        async with serve(server._handler, "127.0.0.1", 0) as listener:
            uri = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
            async with connect(uri) as first:
                assert msgpack_numpy.unpackb(await first.recv()) == META
                async with connect(uri) as second:
                    assert msgpack_numpy.unpackb(await second.recv())["error"] == "active_attempt"
                for obs in (reset(), batch(), infer()):
                    result = await rpc(first, obs)
                    assert result["eval_rpc"] == obs["eval_rpc"]
                released = await rpc(first, {"release_attempt": True, "eval_rpc": {"episode_nonce": "episode"}})
                assert released["release_attempt_finished"] and released["quiescent"]
            async with connect(uri) as next_client:
                assert msgpack_numpy.unpackb(await next_client.recv()) == META
                assert (await rpc(next_client, {"release_attempt": True, "eval_rpc": {"episode_nonce": "unused-attempt"}}))["quiescent"]
        assert policy.calls == ["reset", "add_buffer", "infer"]
    asyncio.run(scenario())


@pytest.mark.parametrize("lost_operation", ["reset", "add_buffer", "infer"])
def test_lost_ack_keeps_attempt_owned_until_resident_restart(lost_operation):
    async def scenario():
        policy = FakePolicy()
        server = WebsocketPolicyServer(policy, metadata=META)
        async with serve(server._handler, "127.0.0.1", 0) as listener:
            uri = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
            async with connect(uri) as first:
                await first.recv()
                for operation, obs in (("reset", reset()), ("add_buffer", batch()), ("infer", infer())):
                    if operation == lost_operation:
                        # 服务端已执行，客户端从未recv回应即断线；不得恢复旧局。
                        policy.done = asyncio.Event()
                        await first.send(msgpack_numpy.Packer().pack(obs))
                        await asyncio.wait_for(policy.done.wait(), 2)
                        break
                    await rpc(first, obs)
            async with connect(uri) as stale:
                assert msgpack_numpy.unpackb(await stale.recv())["error"] == "active_attempt"
        assert policy.calls.count(lost_operation) == 1
    asyncio.run(scenario())


def test_bad_rpc_does_not_invoke_policy():
    async def scenario():
        policy = FakePolicy()
        server = WebsocketPolicyServer(policy, metadata=META)
        async with serve(server._handler, "127.0.0.1", 0) as listener:
            async with connect(f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}") as socket:
                await socket.recv()
                bad = reset()
                bad["eval_rpc"]["expected_server_nonce"] = "wrong"
                result = await rpc(socket, bad)
                assert result["error"] == "contract_violation" and result["rpc_executed"] is False
        assert policy.calls == []
    asyncio.run(scenario())
