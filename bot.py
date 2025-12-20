#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMRunFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transcriptions.language import Language

load_dotenv(override=True)

# Setup OpenTelemetry tracing (always enabled if dependencies are available)
V2V_TRACER = None
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from pipecat.utils.tracing.setup import setup_tracing

    # Initialize the OTLP exporter for pipecat default tracing
    exporter = OTLPSpanExporter(
        endpoint="http://localhost:4317",
        insecure=True,
    )

    # Set up default pipecat tracing
    setup_tracing(
        service_name="pipecat-quickstart-bot",
        exporter=exporter,
    )
    logger.info("OpenTelemetry tracing enabled")

    # Setup separate tracer for v2v speaking events with service name "pipecat_v2v"
    v2v_resource = Resource.create({
        "service.name": "pipecat_v2v",
        "service.instance.id": os.getenv("HOSTNAME", "unknown"),
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })
    v2v_tracer_provider = TracerProvider(resource=v2v_resource)
    v2v_exporter = OTLPSpanExporter(
        endpoint="http://localhost:4317",
        insecure=True,
    )
    v2v_tracer_provider.add_span_processor(BatchSpanProcessor(v2v_exporter))
    V2V_TRACER = v2v_tracer_provider.get_tracer("pipecat_v2v")
    logger.info("V2V OpenTelemetry tracer initialized with service name 'pipecat_v2v'")

except ImportError as e:
    logger.warning(f"Failed to import tracing dependencies: {e}. Tracing will be disabled.")
except Exception as e:
    logger.warning(f"Failed to setup tracing: {e}. Tracing will be disabled.")


class V2VSpeakingTracer(FrameProcessor):
    """Custom FrameProcessor that creates OpenTelemetry spans for speaking events.
    
    Creates spans for:
    - Conversation (parent/root span)
    - User started/stopped speaking (child of conversation)
    - Bot started/stopped speaking (child of conversation)
    
    User and bot speaking spans can overlap within the same parent conversation span.
    Uses service name "pipecat_v2v" for the spans.
    """

    def __init__(self, tracer=None, **kwargs):
        super().__init__(**kwargs)
        self._tracer = tracer
        self._conversation_span = None
        self._conversation_context = None
        self._user_speaking_span = None
        self._bot_speaking_span = None
        self._user_speaking_start_time: Optional[float] = None
        self._bot_speaking_start_time: Optional[float] = None
        self._conversation_started = False

    def start_conversation(self):
        """Start the parent conversation span."""
        if self._tracer and not self._conversation_started:
            self._conversation_span = self._tracer.start_span(
                "conversation",
                attributes={
                    "event.type": "conversation",
                    "conversation.start_time": time.time(),
                }
            )
            # Store the context for creating child spans
            from opentelemetry import trace
            self._conversation_context = trace.set_span_in_context(self._conversation_span)
            self._conversation_started = True
            logger.debug("V2V Tracer: Conversation span started")

    def end_conversation(self):
        """End the parent conversation span."""
        if self._conversation_span is not None:
            # End any active speaking spans first
            if self._user_speaking_span is not None:
                self._handle_user_stopped_speaking()
            if self._bot_speaking_span is not None:
                self._handle_bot_stopped_speaking()
            
            self._conversation_span.set_attribute("conversation.end_time", time.time())
            self._conversation_span.end()
            self._conversation_span = None
            self._conversation_context = None
            self._conversation_started = False
            logger.debug("V2V Tracer: Conversation span ended")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._tracer:
            if isinstance(frame, UserStartedSpeakingFrame):
                self._handle_user_started_speaking()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._handle_user_stopped_speaking()
            elif isinstance(frame, BotStartedSpeakingFrame):
                self._handle_bot_started_speaking()
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._handle_bot_stopped_speaking()

        # Always pass the frame through
        await self.push_frame(frame, direction)

    def _handle_user_started_speaking(self):
        """Create a span when user starts speaking (child of conversation)."""
        if self._user_speaking_span is None and self._conversation_context is not None:
            self._user_speaking_start_time = time.time()
            self._user_speaking_span = self._tracer.start_span(
                "user_speaking",
                context=self._conversation_context,
                attributes={
                    "event.type": "user_speaking",
                    "speaking.start_time": self._user_speaking_start_time,
                }
            )
            logger.debug("V2V Tracer: User started speaking span created")

    def _handle_user_stopped_speaking(self):
        """End the span when user stops speaking."""
        if self._user_speaking_span is not None:
            duration = time.time() - self._user_speaking_start_time if self._user_speaking_start_time else 0
            self._user_speaking_span.set_attribute("speaking.duration_seconds", duration)
            self._user_speaking_span.set_attribute("speaking.end_time", time.time())
            self._user_speaking_span.end()
            self._user_speaking_span = None
            self._user_speaking_start_time = None
            logger.debug(f"V2V Tracer: User stopped speaking span ended (duration: {duration:.2f}s)")

    def _handle_bot_started_speaking(self):
        """Create a span when bot starts speaking (child of conversation)."""
        if self._bot_speaking_span is None and self._conversation_context is not None:
            self._bot_speaking_start_time = time.time()
            self._bot_speaking_span = self._tracer.start_span(
                "bot_speaking",
                context=self._conversation_context,
                attributes={
                    "event.type": "bot_speaking",
                    "speaking.start_time": self._bot_speaking_start_time,
                }
            )
            logger.debug("V2V Tracer: Bot started speaking span created")

    def _handle_bot_stopped_speaking(self):
        """End the span when bot stops speaking."""
        if self._bot_speaking_span is not None:
            duration = time.time() - self._bot_speaking_start_time if self._bot_speaking_start_time else 0
            self._bot_speaking_span.set_attribute("speaking.duration_seconds", duration)
            self._bot_speaking_span.set_attribute("speaking.end_time", time.time())
            self._bot_speaking_span.end()
            self._bot_speaking_span = None
            self._bot_speaking_start_time = None
            logger.debug(f"V2V Tracer: Bot stopped speaking span ended (duration: {duration:.2f}s)")


def load_system_prompt() -> str:
    """Load system prompt from system_prompt.txt file."""
    prompt_file = Path(__file__).parent / "system_prompt.txt"
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_voice_cloning_key() -> str:
    """Load voice cloning key from voice_cloning_key.txt file."""
    key_file = Path(__file__).parent / "voice_cloning_key.txt"
    try:
        with open(key_file, "r", encoding="utf-8") as f:
            key = f.read().strip()
            return key if key else None
    except FileNotFoundError:
        return None

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-transcribe",
        language=Language.HI_IN,
    )

    # Google Cloud TTS with Chirp 3 voice cloning
    voice_cloning_key = load_voice_cloning_key()
    
    # Read Google credentials from file path
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    credentials_content = None
    if credentials_path:
        try:
            with open(credentials_path, "r", encoding="utf-8") as f:
                credentials_content = f.read()
        except FileNotFoundError:
            logger.warning(f"Google credentials file not found: {credentials_path}")
    
    tts_kwargs = {
        "voice_id": "en-US-Chirp3-HD-Charon",
        "params": GoogleTTSService.InputParams(language=Language.EN_US),
    }
    
    if credentials_content:
        tts_kwargs["credentials"] = credentials_content
    
    # Add voice cloning key if available
    if voice_cloning_key:
        tts_kwargs["voice_cloning_key"] = voice_cloning_key
    
    tts = GoogleTTSService(**tts_kwargs)

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": load_system_prompt(),
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # Create V2V speaking tracer for OpenTelemetry spans
    v2v_tracer = V2VSpeakingTracer(tracer=V2V_TRACER)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            v2v_tracer,  # V2V speaking event tracer (captures user/bot speaking events)
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )
    
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        enable_tracing=True,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Start the conversation span for V2V tracing
        v2v_tracer.start_conversation()
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        # End the conversation span for V2V tracing
        v2v_tracer.end_conversation()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()