# For python 3.11 (which is used for the raspberry pi), I was not able to use the newest version of the python-homewizard-energy.
# Therefore, to be able to still communicate with a Plug-In battery, I modified the existing v2/__init__.py from the github:
# https://github.com/homewizard/python-homewizard-energy/tree/main.
# Licensed under the Apache License, Version 2.0


"""Representation of a HomeWizard Energy device."""

from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import Callable, Coroutine
from http import HTTPStatus
from typing import Any, TypeVar

import async_timeout
import backoff
from aiohttp.client import (
    ClientError,
    ClientResponseError,
    ClientSession,
    ClientTimeout,
    TCPConnector,
)
from aiohttp.hdrs import METH_DELETE, METH_GET, METH_POST, METH_PUT
from mashumaro.exceptions import InvalidFieldValue, MissingField

from .const_wrapper import LOGGER
from homewizard_energy.errors import (
    DisabledError,
    NotFoundError,
    RequestError,
    UnsupportedError,
)
from .errors_wrapper import InvalidUserNameError, ResponseError, UnauthorizedError

from homewizard_energy import HomeWizardEnergy
from .models_wrapper import (
    Device,
    Measurement,
    System,
    SystemUpdate,
    Token,
)
from .cacert import CACERT

T = TypeVar("T")


def authorized_method(
    func: Callable[..., Coroutine[Any, Any, T]],
) -> Callable[..., Coroutine[Any, Any, T]]:
    """Decorator method to check if token is set."""

    async def wrapper(self, *args, **kwargs) -> T:
        # pylint: disable=protected-access
        if self._token is None:
            raise UnauthorizedError("Token missing")

        return await func(self, *args, **kwargs)

    return wrapper


# pylint: disable=abstract-method
class HomeWizardEnergyV2(HomeWizardEnergy):
    """Communicate with a HomeWizard Energy device."""

    _ssl: ssl.SSLContext | bool = False
    _identifier: str | None = None
    _lock: asyncio.Lock

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        host: str,
        identifier: str | None = None,
        token: str | None = None,
        clientsession: ClientSession = None,
        timeout: int = 10,
    ):
        """Create a HomeWizard Energy object.

        Args:
            host: IP or URL for device.
            id: ID for device.
            token: Token for device.
            timeout: Request timeout in seconds.
        """
        super().__init__(host, clientsession, timeout)
        self._identifier = identifier
        self._token = token
        self._close_session = clientsession is None

        self._lock = asyncio.Lock()

    async def _create_clientsession(self) -> None:
        """Create a client session."""

        LOGGER.debug("Creating clientsession")

        if self._session is not None:
            raise RuntimeError("Session already exists")

        connector = TCPConnector(
            enable_cleanup_closed=True,
            limit_per_host=1,
        )

        self._close_session = True

        self._session = ClientSession(
            connector=connector, timeout=ClientTimeout(total=self._request_timeout)
        )

    @authorized_method
    async def device(self, reset_cache: bool = False) -> Device:
        """Return the device object."""
        if self._device is not None and not reset_cache:
            return self._device

        _, response = await self._request("/api")
        device = Device.from_json(response)

        # Cache device object
        self._device = device
        return device

    @authorized_method
    async def measurement(self) -> Measurement:
        """Return the measurement object."""
        _, response = await self._request("/api/measurement")
        measurement = Measurement.from_json(response)

        return measurement

    @authorized_method
    async def system(
        self,
        cloud_enabled: bool | None = None,
        status_led_brightness_pct: int | None = None,
        api_v1_enabled: bool | None = None,
    ) -> System:
        """Return the system object."""

        if (
            cloud_enabled is not None
            or status_led_brightness_pct is not None
            or api_v1_enabled is not None
        ):
            data = SystemUpdate(
                cloud_enabled=cloud_enabled,
                status_led_brightness_pct=status_led_brightness_pct,
                api_v1_enabled=api_v1_enabled,
            ).to_dict()
            status, response = await self._request(
                "/api/system", method=METH_PUT, data=data
            )

        else:
            status, response = await self._request("/api/system")

        if status != HTTPStatus.OK:
            error = json.loads(response).get("error", response)
            raise RequestError(f"Failed to get system: {error}")

        system = System.from_json(response)
        return system

    @authorized_method
    async def identify(
        self,
    ) -> None:
        """Send identify request."""
        await self._request("/api/system/identify", method=METH_PUT)

    async def get_token(
        self,
        name: str,
    ) -> str:
        """Get authorization token from device."""
        status, response = await self._request(
            "/api/user", method=METH_POST, data={"name": f"local/{name}"}
        )

        if status == HTTPStatus.FORBIDDEN:
            raise DisabledError("User creation is not enabled on the device")

        if status != HTTPStatus.OK:
            error = json.loads(response).get("error", response)
            raise InvalidUserNameError(
                f"Error occurred while getting token: {error}", error
            )

        try:
            token = Token.from_json(response).token
        except (InvalidFieldValue, MissingField) as ex:
            raise ResponseError("Failed to get token") from ex

        self._token = token
        return token

    async def _get_ssl_context(self) -> ssl.SSLContext:
        """
        Get a clientsession that is tuned for communication with the HomeWizard Energy Device
        """

        def _build_ssl_context() -> ssl.SSLContext:
            context = ssl.create_default_context(cadata=CACERT)
            context.verify_flags = ssl.VERIFY_X509_PARTIAL_CHAIN  # pylint: disable=no-member
            if self._identifier is not None:
                context.hostname_checks_common_name = True
            else:
                context.check_hostname = False  # Skip hostname validation
                context.verify_mode = ssl.CERT_REQUIRED  # Keep SSL verification active
            return context

        # Creating an SSL context has some blocking IO so need to run it in the executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _build_ssl_context)

    @backoff.on_exception(backoff.expo, RequestError, max_tries=3, logger=None)
    async def _request(
        self, path: str, method: str = METH_GET, data: object = None
    ) -> tuple[HTTPStatus, dict[str, Any] | None]:
        """Make a request to the API."""

        async with self._lock:
            if self._session is None:
                await self._create_clientsession()

        async with self._lock:
            if self._ssl is False:
                self._ssl = await self._get_ssl_context()

        # Construct request
        url = f"https://{self.host}{path}"
        headers = {
            "Content-Type": "application/json",
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"

        LOGGER.debug("%s, %s, %s", method, url, data)

        try:
            async with async_timeout.timeout(self._request_timeout):
                async with self._lock:
                    resp = await self._session.request(
                        method,
                        url,
                        json=data,
                        headers=headers,
                        ssl=self._ssl,
                        server_hostname=self._identifier,
                    )
                LOGGER.debug("%s, %s", resp.status, await resp.text("utf-8"))
        except asyncio.TimeoutError as exception:
            raise RequestError(
                f"Timeout occurred while connecting to the HomeWizard Energy device at {self.host}"
            ) from exception
        except (ClientError, ClientResponseError) as exception:
            raise RequestError(
                f"Error occurred while communicating with the HomeWizard Energy device at {self.host}"
            ) from exception

        match resp.status:
            case HTTPStatus.UNAUTHORIZED:
                raise UnauthorizedError("Token rejected")
            case HTTPStatus.NO_CONTENT:
                # No content, just return
                return (HTTPStatus.NO_CONTENT, None)
            case HTTPStatus.NOT_FOUND:
                raise NotFoundError("Resource not found")
            case HTTPStatus.OK:
                pass

        return (resp.status, await resp.text())

    async def __aenter__(self) -> HomeWizardEnergyV2:
        """Async enter.

        Returns:
            The HomeWizardEnergyV2 object.
        """
        return self
