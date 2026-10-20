import socket
import asyncio
import json
import argparse
import math
from dataclasses import asdict, dataclass
from itertools import count
from typing import Any

_request_ids = count(1)


class InvalidMarstekResponseError(ValueError):
    """Raised when a Marstek response cannot be parsed as ES status."""


@dataclass(frozen=True)
class MarstekESStatus:
    bat_soc: int | float
    bat_cap: int | float
    pv_power: int | float
    ongrid_power: int | float
    offgrid_power: int | float
    total_pv_energy: int | float
    total_grid_output_energy: int | float
    total_grid_input_energy: int | float
    total_load_energy: int | float


def get_marstek_es_status_message():
    """
    The ES (Energy System) component contains the device's basic power information
    and energy statistics, and can configure or monitor the device's operating
    status.
    --> 1. ES.GetStatus: Query the device's basic electrical energy information.
    2. ES.SetMode: Configure the device's operating mode.
    3. ES.GetMode: Get information about the operating mode of the device.
    """
    return to_marstek_message("ES.GetStatus", {"id": 0})


def get_marstek_bat_status_message():
    return to_marstek_message("Bat.GetStatus", {"id": 0})


def to_marstek_message(method: str, params: dict[str, Any]) -> bytes:
    request_dict = {"id": 0, "method": method, "params": params}
    # 1. Convert dict -> JSON string:
    # '{"id": 1, "method": "PV.GetStatus", "params": {"id": 0}}'
    json_string = json.dumps(request_dict)

    # 2. Convert JSON string -> raw bytes type: b'{"id": 1, ...}'
    message_bytes = json_string.encode("utf-8")
    return message_bytes


async def request_response_udp(
    message: bytes, target_ip: str, target_port: int
) -> bytes | None:
    loop = asyncio.get_running_loop()
    request = json.loads(message)
    request["id"] = next(_request_ids)
    message = json.dumps(request).encode("utf-8")

    # Creating the socket inside the function guarantees a unique local port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)

    try:
        # Send to the specific destination
        await loop.sock_sendto(sock, message, (target_ip, target_port))

        # Wait safely for the response. No other function can steal this packet
        # because it is tied directly to this specific 'sock' instance.
        data, sender_addr = await asyncio.wait_for(
            loop.sock_recvfrom(sock, 1024), timeout=3.0
        )
        return data

    except asyncio.TimeoutError:
        print(f"Timed out waiting for {target_ip}:{target_port}")
        return None
    finally:
        sock.close()  # Always clean up the socket resource


def parse_marstek_es_status(response: bytes) -> MarstekESStatus:
    try:
        response_data = json.loads(response)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InvalidMarstekResponseError(
            "Marstek ES status response is not valid JSON"
        ) from error

    if not isinstance(response_data, dict):
        raise InvalidMarstekResponseError(
            "Marstek ES status response must be a JSON object"
        )

    status_data = response_data.get("result", response_data)
    if not isinstance(status_data, dict):
        raise InvalidMarstekResponseError(
            "Marstek ES status result must be a JSON object"
        )

    required_fields = (
        "bat_soc",
        "bat_cap",
        "pv_power",
        "ongrid_power",
        "offgrid_power",
        "total_pv_energy",
        "total_grid_output_energy",
        "total_grid_input_energy",
        "total_load_energy",
    )
    missing_fields = [field for field in required_fields if field not in status_data]
    if missing_fields:
        raise InvalidMarstekResponseError(
            f"Marstek ES status result is missing fields: {', '.join(missing_fields)}"
        )

    values: dict[str, int | float] = {}
    for field in required_fields:
        value = status_data[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise InvalidMarstekResponseError(
                f"Marstek ES status field {field!r} must be a finite number"
            )
        values[field] = value

    return MarstekESStatus(**values)


async def request_marstek_es_status(
    target_ip: str, target_port: int
) -> MarstekESStatus | None:
    response = await request_response_udp(
        get_marstek_es_status_message(), target_ip, target_port
    )
    if response is None:
        return None
    return parse_marstek_es_status(response)


class MarstekUDPClient:
    def __init__(self, host: str, port: int = 50000):
        self.host = host
        self.port = port

    async def __aenter__(self) -> "MarstekUDPClient":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None

    async def es_status(self) -> MarstekESStatus | None:
        return await request_marstek_es_status(self.host, self.port)


async def send_status_request(target_ip: str, target_port: int) -> int:
    failed = False

    print("Requesting ES status...")
    try:
        es_status = await request_marstek_es_status(target_ip, target_port)
    except InvalidMarstekResponseError as error:
        print(f"Invalid ES status response: {error}")
        return 1

    if es_status is None:
        failed = True
    else:
        print(f"ES status response: {json.dumps(asdict(es_status), indent=2)}")

    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send a status request to a Marstek Venus."
    )
    parser.add_argument("ip", help="IP address of the Marstek Venus")
    parser.add_argument(
        "--port", type=int, default=30000, help="UDP port (default: 30000)"
    )
    args = parser.parse_args()
    return asyncio.run(send_status_request(args.ip, args.port))


if __name__ == "__main__":
    raise SystemExit(main())
