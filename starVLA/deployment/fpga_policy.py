"""CPU preprocessing frontend for the VLA-JEPA FPGA TCP service.

This module intentionally does not import or construct any VLA model class.  It
only performs the tokenizer/image-processor work that precedes the exported
accelerator graph, creates the Flow Matching noise input, and transports the
six protocol-v2 fields to the FPGA service.
"""

from __future__ import annotations

import socket
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


REQUEST_MAGIC = b"VLARQ002"
RESPONSE_MAGIC = b"VLARS002"
HEALTH_REQUEST = b"VLAPING1"
HEALTH_RESPONSE = b"VLAPONG1"
PROTOCOL_VERSION = 2

IMAGE_TOKEN_ID = 151655
EMBODIED_ACTION_TOKEN_ID = 151697
ACTION_TOKEN_0_ID = 151669
EMBEDDING_VOCAB_SIZE = 151936
ACTION_DIM = 7
ACTION_HORIZON = 7
DEFAULT_INFERENCE_SEED = 42
IMAGE_SIZE = 224
MIN_LANGUAGE_TOKENS = 134
MAX_LANGUAGE_TOKENS = 512
EXPECTED_PIXEL_VALUES_SHAPE = (512, 1536)
EXPECTED_IMAGE_GRID = np.asarray([[1, 16, 16], [1, 16, 16]], dtype=np.int64)

ACTION_TOKEN_TEMPLATE = "<|action_{}|>"
EMBODIED_ACTION_TOKEN = "<|embodied_action|>"
PROMPT_TEMPLATE = (
    "Your task is {instruction}. Infer the temporal dynamics from frames "
    "{actions} and produce the corresponding policy actions {e_actions}."
)


class FPGATransportError(RuntimeError):
    """The FPGA endpoint rejected or could not complete a request."""


def _read_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise FPGATransportError(
                f"FPGA server closed the connection with {remaining} byte(s) pending"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _FPGAV2Transport:
    """One-request-per-connection implementation of the binary v2 protocol."""

    def __init__(self, host: str, port: int, timeout: float) -> None:
        if not host:
            raise ValueError("FPGA host must not be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("FPGA port must be in 1..65535")
        if timeout <= 0:
            raise ValueError("FPGA timeout must be positive")
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def health_check(self) -> None:
        probe_timeout = min(self.timeout, 10.0)
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=probe_timeout
            ) as connection:
                connection.settimeout(probe_timeout)
                connection.sendall(HEALTH_REQUEST)
                response = _read_exact(connection, len(HEALTH_RESPONSE))
        except (OSError, FPGATransportError) as error:
            raise FPGATransportError(
                f"FPGA server is unavailable at {self.host}:{self.port}"
            ) from error
        if response != HEALTH_RESPONSE:
            raise FPGATransportError("FPGA server returned an invalid health response")

    def infer(
        self,
        instruction: str,
        pixel_values: np.ndarray,
        input_ids: np.ndarray,
        image_grid_thw: np.ndarray,
        normalized_state: np.ndarray,
        initial_actions: np.ndarray,
    ) -> np.ndarray:
        fields = [
            instruction.encode("utf-8"),
            np.asarray(pixel_values, dtype="<f2").tobytes(order="C"),
            np.asarray(input_ids, dtype="<i8").tobytes(order="C"),
            np.asarray(image_grid_thw, dtype="<i8").tobytes(order="C"),
            np.asarray(normalized_state, dtype="<f2").tobytes(order="C"),
            np.asarray(initial_actions, dtype="<f2").tobytes(order="C"),
        ]
        header = REQUEST_MAGIC + struct.pack(
            "!7I", PROTOCOL_VERSION, *(len(field) for field in fields)
        )

        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            ) as connection:
                connection.settimeout(self.timeout)
                connection.sendall(header)
                for field in fields:
                    connection.sendall(field)

                magic = _read_exact(connection, len(RESPONSE_MAGIC))
                if magic != RESPONSE_MAGIC:
                    raise FPGATransportError("FPGA server returned invalid response magic")
                version, status, message_size, payload_size = struct.unpack(
                    "!4I", _read_exact(connection, 16)
                )
                if version != PROTOCOL_VERSION:
                    raise FPGATransportError(
                        f"FPGA server returned protocol version {version}"
                    )
                message = _read_exact(connection, message_size).decode(
                    "utf-8", "replace"
                )
                payload = _read_exact(connection, payload_size)
        except FPGATransportError:
            raise
        except OSError as error:
            raise FPGATransportError(
                f"FPGA request failed at {self.host}:{self.port}"
            ) from error

        if status != 0:
            raise FPGATransportError(
                f"FPGA server rejected inference (status={status}): {message}"
            )
        if len(payload) != ACTION_HORIZON * ACTION_DIM * 2:
            raise FPGATransportError(
                f"FPGA response has {len(payload)} bytes; expected 98"
            )
        actions = np.frombuffer(payload, dtype="<f2").astype(np.float32)
        actions = actions.reshape(ACTION_HORIZON, ACTION_DIM)
        if not np.isfinite(actions).all():
            raise FPGATransportError("FPGA response contains non-finite actions")
        return actions


class VLAJEPAFPGAPolicy:
    """Drop-in inference policy backed by the FPGA server.

    ``predict_action`` accepts the same observation arguments used by the
    original VLA-JEPA policy, but batch size is deliberately fixed to one to
    match the serialized hardware service.
    """

    def __init__(
        self,
        processor_path: str | Path,
        *,
        fpga_host: str = "127.0.0.1",
        fpga_port: int = 18080,
        timeout: float = 3600.0,
        local_files_only: bool = True,
        check_health: bool = True,
        inference_seed: int = DEFAULT_INFERENCE_SEED,
    ) -> None:
        try:
            from transformers import AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "transformers with Qwen3-VL support is required for preprocessing"
            ) from error

        processor_path = Path(processor_path).expanduser().resolve()
        if local_files_only and not processor_path.is_dir():
            raise FileNotFoundError(f"processor directory does not exist: {processor_path}")
        self.processor = AutoProcessor.from_pretrained(
            str(processor_path), local_files_only=local_files_only
        )
        self.processor.tokenizer.padding_side = "left"
        self._install_and_validate_tokens()
        self.inference_seed = int(inference_seed)
        self._action_generator = torch.Generator(device="cpu")
        self._action_generator.manual_seed(self.inference_seed)
        self._transport = _FPGAV2Transport(fpga_host, fpga_port, timeout)
        if check_health:
            self._transport.health_check()

        self.action_dim = ACTION_DIM
        self.action_horizon = ACTION_HORIZON
        self.image_size = IMAGE_SIZE
        # Preserve the small part of the historical policy surface consumed by
        # MetaRoboArm and external metadata helpers.
        self.config = SimpleNamespace(
            framework=SimpleNamespace(
                action_model=SimpleNamespace(
                    action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON
                )
            ),
            datasets=SimpleNamespace(
                vla_data=SimpleNamespace(resolution_size=IMAGE_SIZE)
            ),
        )

    def _install_and_validate_tokens(self) -> None:
        tokenizer = self.processor.tokenizer
        for index in range(ACTION_HORIZON * 4):
            token = ACTION_TOKEN_TEMPLATE.format(index)
            if token not in tokenizer.get_vocab():
                tokenizer.add_tokens([token], special_tokens=True)
        if EMBODIED_ACTION_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_tokens([EMBODIED_ACTION_TOKEN], special_tokens=True)

        action_token_id = tokenizer.convert_tokens_to_ids(
            ACTION_TOKEN_TEMPLATE.format(0)
        )
        embodied_token_id = tokenizer.convert_tokens_to_ids(EMBODIED_ACTION_TOKEN)
        if action_token_id != ACTION_TOKEN_0_ID:
            raise RuntimeError(
                f"processor action token id is {action_token_id}; FPGA expects "
                f"{ACTION_TOKEN_0_ID}"
            )
        if embodied_token_id != EMBODIED_ACTION_TOKEN_ID:
            raise RuntimeError(
                f"processor embodied-action token id is {embodied_token_id}; FPGA "
                f"expects {EMBODIED_ACTION_TOKEN_ID}"
            )
        if len(tokenizer) > EMBEDDING_VOCAB_SIZE:
            raise RuntimeError("processor vocabulary exceeds the FPGA embedding table")

    @staticmethod
    def _prompt(instruction: str) -> str:
        return PROMPT_TEMPLATE.format(
            instruction=instruction,
            actions=ACTION_TOKEN_TEMPLATE.format(0) * 8,
            e_actions=EMBODIED_ACTION_TOKEN * 32,
        )

    def prepare_inputs(
        self, images: Sequence[Image.Image], instruction: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(images) != 2:
            raise ValueError("FPGA inference requires exactly two images")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        prepared_images: list[Image.Image] = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("images must contain PIL.Image values")
            prepared_images.append(
                image.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                )
            )

        message = [{
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in prepared_images),
                {"type": "text", "text": self._prompt(instruction.strip())},
            ],
        }]
        batch = self.processor.apply_chat_template(
            [message],
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            input_ids = batch["input_ids"].detach().cpu().numpy().astype(np.int64)
            pixel_values = (
                batch["pixel_values"].detach().cpu().numpy().astype(np.float16)
            )
            image_grid_thw = (
                batch["image_grid_thw"].detach().cpu().numpy().astype(np.int64)
            )
        except (AttributeError, KeyError) as error:
            raise RuntimeError("Qwen3-VL processor returned an invalid tensor set") from error

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError(f"unexpected input_ids shape: {input_ids.shape}")
        input_ids = input_ids[0]
        token_count = input_ids.size
        if not MIN_LANGUAGE_TOKENS <= token_count <= MAX_LANGUAGE_TOKENS:
            raise ValueError(
                f"instruction produces {token_count} tokens; FPGA supports "
                f"{MIN_LANGUAGE_TOKENS}..{MAX_LANGUAGE_TOKENS}"
            )
        if pixel_values.shape != EXPECTED_PIXEL_VALUES_SHAPE:
            raise RuntimeError(
                f"unexpected pixel_values shape {pixel_values.shape}; expected "
                f"{EXPECTED_PIXEL_VALUES_SHAPE}"
            )
        if image_grid_thw.shape != EXPECTED_IMAGE_GRID.shape or not np.array_equal(
            image_grid_thw, EXPECTED_IMAGE_GRID
        ):
            raise RuntimeError(
                f"unexpected image_grid_thw {image_grid_thw.tolist()}; expected "
                f"{EXPECTED_IMAGE_GRID.tolist()}"
            )
        if int(np.count_nonzero(input_ids == IMAGE_TOKEN_ID)) != 128:
            raise RuntimeError("processor output must contain exactly 128 image tokens")
        if int(np.count_nonzero(input_ids == EMBODIED_ACTION_TOKEN_ID)) != 32:
            raise RuntimeError(
                "processor output must contain exactly 32 embodied-action tokens"
            )
        if int(np.count_nonzero(input_ids == ACTION_TOKEN_0_ID)) != 8:
            raise RuntimeError("processor output must contain exactly 8 action-0 tokens")
        if np.any(input_ids < 0) or np.any(input_ids >= EMBEDDING_VOCAB_SIZE):
            raise RuntimeError("processor emitted a token outside the FPGA vocabulary")
        return pixel_values, input_ids, image_grid_thw

    def initial_actions(self) -> np.ndarray:
        """Draw the next Flow Matching noise chunk from this policy's RNG stream."""
        return (
            torch.randn(
                (ACTION_HORIZON, ACTION_DIM),
                dtype=torch.float32,
                device="cpu",
                generator=self._action_generator,
            )
            .to(torch.float16)
            .numpy()
        )

    def predict_action(
        self,
        *,
        batch_images: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str],
        state: Any = None,
        inference_seed: int | None = None,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        if inference_seed is not None and int(inference_seed) != self.inference_seed:
            raise ValueError(
                "inference_seed is fixed when VLAJEPAFPGAPolicy is constructed; "
                "create a new policy instance to use a different seed"
            )
        if len(batch_images) != 1 or len(instructions) != 1:
            raise ValueError("FPGA policy only supports batch size one")
        normalized_state = np.asarray(state, dtype=np.float32).reshape(-1)
        if normalized_state.shape != (ACTION_DIM,):
            raise ValueError("normalized state must contain seven values")
        if not np.isfinite(normalized_state).all():
            raise ValueError("normalized state contains non-finite values")

        pixel_values, input_ids, image_grid_thw = self.prepare_inputs(
            batch_images[0], instructions[0]
        )
        actions = self._transport.infer(
            instructions[0].strip(),
            pixel_values,
            input_ids,
            image_grid_thw,
            normalized_state.astype(np.float16),
            self.initial_actions(),
        )
        return {"normalized_actions": actions[None, ...]}
