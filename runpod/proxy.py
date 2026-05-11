import asyncio
import os
import socket
import threading
import time
import uuid
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request

count = 0
prefill_instances: dict[str, Any] = {}
decode_instances: dict[str, Any] = {}

prefill_cv = threading.Condition()
decode_cv = threading.Condition()

DEFAULT_PING_SECONDS = 5


def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"🔴Remove [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["type"] == "P":
                global prefill_instances, prefill_cv
                with prefill_cv:
                    node = prefill_instances.get(data["http_address"], None)
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(prefill_instances)
            elif data["type"] == "D":
                global decode_instances, decode_cv
                with decode_cv:
                    node = decode_instances.get(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)
            else:
                print(f"Unexpected message from {remote_address}: {data}")
                return

            if node is None:
                print(f"🔵Add [HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}]")


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    t = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    t.start()
    return t


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


async def collect_response(url, data, request_id):
    """Send request and fully consume the response (used for prefill)."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "X-Request-Id": request_id,
    }
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=url, json=data, headers=headers) as response:
                await response.read()
                print(f"prefill response status: {response.status}")
                return response.status
    except Exception as e:
        print(f"collect_response error ({url}): {e}")


async def stream_response(url, data, request_id, queue: asyncio.Queue):
    """Send request and stream chunks into a queue (used for decode)."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "X-Request-Id": request_id,
    }
    try:
        print(f"[DECODE] About to POST to {url}")
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            print(f"[DECODE] Session opened, sending POST")
            async with session.post(url=url, json=data, headers=headers) as response:
                print(f"[DECODE] Got response headers, status={response.status}")
                async for chunk in response.content.iter_chunked(1024):
                    print(f"[DECODE] Got chunk of size {len(chunk)}")
                    await queue.put(chunk)
                print(f"[DECODE] Stream ended normally")
    except Exception as e:
        print(f"[DECODE] Exception: {type(e).__name__}: {e}")
    finally:
        await queue.put(None)  # sentinel — always unblocks generate()


async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        global count, prefill_instances, prefill_cv, decode_instances, decode_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
            prefill_zmq_addr = prefill_zmq_addr[0]

        with decode_cv:
            decode_list = list(decode_instances.items())
            decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_zmq_addr = decode_zmq_addr[0]

        print(
            f"handle_request count: {count}, [HTTP:{prefill_addr}, "
            f"ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, ZMQ:{decode_zmq_addr}]"
        )
        count += 1

        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
            f"{decode_zmq_addr}_{random_uuid()}"
        )

        queue: asyncio.Queue = asyncio.Queue()

        # Launch prefill and decode concurrently so decode is listening
        # when prefill tries to send KV cache via NCCL
        prefill_task = asyncio.create_task(
            collect_response(
                f"http://{prefill_addr}{request.path}", prefill_request, request_id
            )
        )
        decode_task = asyncio.create_task(
            stream_response(
                f"http://{decode_addr}{request.path}",
                original_request_data,
                request_id,
                queue,
            )
        )

        async def generate():
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
            await prefill_task
            await decode_task

        response = await make_response(generate())
        response.timeout = None
        return response

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error in proxy:")
        print("".join(traceback.format_exception(*exc_info)))


app.add_url_rule("/v1/completions", view_func=handle_request, methods=["POST"])
app.add_url_rule("/v1/chat/completions", view_func=handle_request, methods=["POST"])

if __name__ == "__main__":
    t = start_service_discovery("0.0.0.0", 30001)
    app.run(host="0.0.0.0", port=10001)
    t.join()
