#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os
import asyncio
import json
import wave
import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    LLMContextFrame,
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
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transcriptions.language import Language
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from async_memory import AsyncConversationMemory, ChatMessage, MessageRole
from tracer import V2VSpeakingTracer
from mem0 import MemoryClient

load_dotenv(override=True)

USER_NAME = "Mayank"

# Create recordings directory
RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)


class ConversationLogger:
    """
    Handles logging of conversation messages to JSON and audio files with turn-based naming.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the conversation logger.
        
        Args:
            session_id: Optional session ID for file naming. If not provided, uses timestamp.
        """
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = RECORDINGS_DIR / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        self.messages: list[dict] = []
        self.user_turn_count = 0
        self.bot_turn_count = 0
        self.json_file = self.session_dir / "conversation.json"
        
        logger.info(f"ConversationLogger initialized. Session: {self.session_id}")
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation log."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        if role == "user":
            self.user_turn_count += 1
            message["turn"] = f"user_turn_{self.user_turn_count}"
        elif role == "assistant":
            self.bot_turn_count += 1
            message["turn"] = f"bot_turn_{self.bot_turn_count}"
        
        self.messages.append(message)
        self._save_json()
        logger.debug(f"ConversationLogger: Added {role} message (turn: {message.get('turn', 'N/A')})")
    
    def _save_json(self):
        """Save messages to JSON file."""
        conversation_data = {
            "session_id": self.session_id,
            "start_time": self.messages[0]["timestamp"] if self.messages else None,
            "last_update": datetime.datetime.now().isoformat(),
            "total_user_turns": self.user_turn_count,
            "total_bot_turns": self.bot_turn_count,
            "messages": self.messages,
        }
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    def save_audio(self, audio_data: bytes, sample_rate: int, num_channels: int, role: str, turn_number: int):
        """Save audio data to a WAV file with turn-based naming."""
        filename = f"{role}_turn_{turn_number}.wav"
        filepath = self.session_dir / filename
        
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        
        logger.info(f"ConversationLogger: Saved audio to {filepath}")
        return filepath
    
    def get_current_user_turn(self) -> int:
        """Get the current user turn number."""
        return self.user_turn_count
    
    def get_current_bot_turn(self) -> int:
        """Get the current bot turn number."""
        return self.bot_turn_count


class AssistantResponseLogger(FrameProcessor):
    """
    A FrameProcessor that captures assistant responses from the LLM.
    
    This processor should be placed AFTER the LLM to capture TextFrames and
    LLMFullResponseEndFrame, logging assistant messages to the ConversationLogger
    and memory.
    """
    
    def __init__(
        self,
        memory: AsyncConversationMemory,
        conversation_logger: Optional[ConversationLogger] = None,
        **kwargs
    ):
        """
        Initialize the assistant response logger.
        
        Args:
            memory: The AsyncConversationMemory instance
            conversation_logger: Optional ConversationLogger for JSON logging
        """
        super().__init__(**kwargs)
        self._memory = memory
        self._conversation_logger = conversation_logger
        
        # Track assistant response aggregation
        self._llm_response_started = False
        self._assistant_aggregation: list[str] = []
        
        # Track pending memory tasks for cleanup
        self._pending_tasks: set[asyncio.Task] = set()
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames to capture assistant responses."""
        await super().process_frame(frame, direction)
        
        # Track LLM response start
        if isinstance(frame, LLMFullResponseStartFrame):
            self._llm_response_started = True
            self._assistant_aggregation = []
        
        # Capture assistant text during response
        elif isinstance(frame, TextFrame) and self._llm_response_started:
            if frame.text:
                self._assistant_aggregation.append(frame.text)
        
        # Capture completed assistant response
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_assistant_response_end()
        
        # Always pass the frame through
        await self.push_frame(frame, direction)
    
    async def _handle_assistant_response_end(self):
        """Handle assistant response completion - log to JSON and add to memory."""
        self._llm_response_started = False
        
        if not self._assistant_aggregation:
            return
        
        # Combine aggregated text
        full_response = "".join(self._assistant_aggregation)
        self._assistant_aggregation = []
        
        if not full_response.strip():
            return
        
        logger.debug(f"AssistantResponseLogger: Capturing assistant message: {full_response[:50]}...")
        
        # Log to conversation JSON
        if self._conversation_logger:
            self._conversation_logger.add_message("assistant", full_response)
        
        # Add to memory in non-blocking way
        message = ChatMessage(role=MessageRole.ASSISTANT, content=full_response)
        task = self._memory.put_nowait(message)
        self._track_task(task)
    
    def _track_task(self, task: asyncio.Task):
        """Track a task and clean up when done."""
        self._pending_tasks.add(task)
        task.add_done_callback(self._task_done)
    
    def _task_done(self, task: asyncio.Task):
        """Remove completed task from tracking."""
        self._pending_tasks.discard(task)
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc:
                logger.error(f"AssistantResponseLogger: Task failed with error: {exc}")


class MemoryContextManager(FrameProcessor):
    """
    A FrameProcessor that integrates AsyncConversationMemory with pipecat's LLMContext.
    
    This processor should be placed AFTER context_aggregator.user() and BEFORE llm.
    It handles user messages and syncs memory to context before LLM processes.
    
    Note: Assistant response capture is handled by AssistantResponseLogger placed AFTER the LLM.
    """
    
    def __init__(
        self,
        memory: AsyncConversationMemory,
        context: LLMContext,
        system_prompt: str,
        conversation_logger: Optional[ConversationLogger] = None,
        **kwargs
    ):
        """
        Initialize the memory context manager.
        
        Args:
            memory: The AsyncConversationMemory instance
            context: The LLMContext to sync with
            system_prompt: The system prompt to always include
            conversation_logger: Optional ConversationLogger for JSON logging
        """
        super().__init__(**kwargs)
        self._memory = memory
        self._context = context
        self._system_prompt = system_prompt
        self._conversation_logger = conversation_logger
        
        # Track last user message to avoid duplicates
        self._last_user_message: Optional[str] = None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames to sync memory with context."""
        await super().process_frame(frame, direction)
        
        # LLMContextFrame is created by context_aggregator.user() and contains user message
        # This is our opportunity to sync memory before LLM processes
        if isinstance(frame, LLMContextFrame):
            await self._handle_llm_context_frame(frame)
        
        # Always pass the frame through
        await self.push_frame(frame, direction)
    
    async def _handle_llm_context_frame(self, frame: LLMContextFrame):
        """
        Handle LLMContextFrame - extract user message, add to memory, sync context.
        
        The context_aggregator.user() has already added the user message to context.
        We extract it, add to memory (which may trigger summarization), 
        then rebuild context from memory.
        """
        # Get current messages from the context (includes the new user message)
        current_messages = self._context.get_messages()
        
        # Find the last user message (just added by aggregator)
        last_user_msg = None
        for msg in reversed(current_messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        # Add user message to memory if it's new
        if last_user_msg and last_user_msg != self._last_user_message:
            self._last_user_message = last_user_msg
            logger.debug(f"MemoryContextManager: Adding user message to memory: {last_user_msg[:50]}...")
            
            # Log to conversation JSON
            if self._conversation_logger:
                self._conversation_logger.add_message("user", last_user_msg)
            
            # Add to memory - await to ensure it's processed before we sync
            user_chat_msg = ChatMessage(role=MessageRole.USER, content=last_user_msg)
            self._memory.put_nowait(user_chat_msg)
        
        # Now sync memory to context
        await self._sync_memory_to_context()
    
    async def _sync_memory_to_context(self):
        """Sync memory state to LLMContext before LLM runs."""
        # Get context from memory (includes system prompt + summary + recent messages)
        memory_context = self._memory.get_context_for_llm(self._system_prompt)
        
        # Update the LLMContext with memory-managed messages
        self._context.set_messages(memory_context)
        
        state = self._memory.get_state()
        logger.debug(
            f"MemoryContextManager: Synced context - "
            f"Messages: {len(memory_context)}, "
            f"Has Summary: {self._memory.has_summary}, "
            f"Summarized: {state.messages_summarized}"
        )
        
        # Log the summary if it exists
        if self._memory.has_summary:
            logger.debug(f"MemoryContextManager: Current summary: {state.summary}")
        
        # Log all messages being inserted into context
        logger.debug(f"MemoryContextManager: Messages in LLM context:")
        for i, msg in enumerate(memory_context):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long content for readability
            content_preview = content[:100] + "..." if len(content) > 100 else content
            logger.debug(f"  [{i}] {role}: {content_preview}")
    
    @property
    def memory(self) -> AsyncConversationMemory:
        """Get the memory instance."""
        return self._memory
    
    @property
    def context(self) -> LLMContext:
        """Get the context instance."""
        return self._context
    
    @property
    def conversation_logger(self) -> Optional[ConversationLogger]:
        """Get the conversation logger instance."""
        return self._conversation_logger

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


def fetch_user_memories(user_id: str) -> str:
    """Fetch user memories from mem0 and format them as a string."""
    mem0_api_key = os.getenv("MEM0_API_KEY")
    if not mem0_api_key:
        logger.warning("MEM0_API_KEY not found in environment. User memories will be empty.")
        return ""
    
    try:
        memory_client = MemoryClient(api_key=mem0_api_key)
        memories = memory_client.get_all(
            filters={
                "AND": [
                    {
                        "user_id": user_id.lower(),
                    }
                ]
            }
        )
        
        if not memories or not memories.get("results"):
            logger.info(f"No memories found for user: {user_id}")
            return ""
        
        # Format memories as a bullet list
        memory_list = [f"- {memory['memory']}" for memory in memories["results"]]
        formatted_memories = "\n".join(memory_list)
        logger.info(f"Fetched {len(memory_list)} memories for user: {user_id}")
        return formatted_memories
    except Exception as e:
        logger.error(f"Failed to fetch user memories: {e}")
        return ""

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

    # Load system prompt and interpolate user info
    user_memories = fetch_user_memories(USER_NAME)
    system_prompt = load_system_prompt().replace("{user_name}", USER_NAME).replace("{user_memories}", user_memories)
    
    # Initialize AsyncConversationMemory for context management
    # Token limit of 500 allows ~25-30 messages before summarization kicks in
    conversation_memory = AsyncConversationMemory(
        api_key=os.getenv("OPENAI_API_KEY"),
        token_limit=500,  # Adjust based on your needs
        model="gpt-4o-mini",  # Fast model for summarization
        auto_summarize=True,
        summarize_prompt="""Summarize the conversation concisely. 
Preserve: names, numbers, key decisions, action items, and important context.
Keep it brief but comprehensive enough to continue the conversation naturally."""
    )
    
    # Initialize LLMContext with just the system prompt
    # The MemoryContextManager will manage messages dynamically
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)
    
    # Create ConversationLogger for JSON logging and audio recording
    conversation_logger = ConversationLogger()
    
    # Create MemoryContextManager to sync AsyncConversationMemory with LLMContext
    # This is placed BEFORE the LLM to handle user messages and sync context
    memory_context_manager = MemoryContextManager(
        memory=conversation_memory,
        context=context,
        system_prompt=system_prompt,
        conversation_logger=conversation_logger,
    )
    
    # Create AssistantResponseLogger to capture assistant responses
    # This is placed AFTER the LLM to capture TextFrames and log them
    assistant_response_logger = AssistantResponseLogger(
        memory=conversation_memory,
        conversation_logger=conversation_logger,
    )
    
    # Create AudioBufferProcessor for turn-based audio recording
    # Enable turn_audio to get individual audio clips for each speaking turn
    audio_buffer = AudioBufferProcessor(
        num_channels=1,  # Mono audio
        enable_turn_audio=True,  # Enable per-turn audio recording
        user_continuous_stream=True,  # User has continuous audio stream
    )
    
    # Track turn counts for audio file naming
    user_audio_turn = 0
    bot_audio_turn = 0
    
    @audio_buffer.event_handler("on_user_turn_audio_data")
    async def on_user_turn_audio(buffer, audio, sample_rate, num_channels):
        """Handle user turn audio data - save with turn-based naming."""
        nonlocal user_audio_turn
        user_audio_turn += 1
        conversation_logger.save_audio(
            audio_data=audio,
            sample_rate=sample_rate,
            num_channels=num_channels,
            role="user",
            turn_number=user_audio_turn,
        )
    
    @audio_buffer.event_handler("on_bot_turn_audio_data")
    async def on_bot_turn_audio(buffer, audio, sample_rate, num_channels):
        """Handle bot turn audio data - save with turn-based naming."""
        nonlocal bot_audio_turn
        bot_audio_turn += 1
        conversation_logger.save_audio(
            audio_data=audio,
            sample_rate=sample_rate,
            num_channels=num_channels,
            role="bot",
            turn_number=bot_audio_turn,
        )

    # Create V2V speaking tracer for OpenTelemetry spans
    v2v_tracer = V2VSpeakingTracer(tracer=V2V_TRACER)

    # Pipeline with MemoryContextManager placed AFTER context_aggregator.user() 
    # and BEFORE llm to intercept LLMContextFrame and sync memory
    # AssistantResponseLogger is placed AFTER llm to capture assistant responses
    # AudioBufferProcessor is placed AFTER transport.output() to capture both user and bot audio
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            v2v_tracer,  # V2V speaking event tracer (after STT to capture TranscriptionFrames)
            context_aggregator.user(),  # User responses - creates LLMContextFrame
            memory_context_manager,  # Intercepts LLMContextFrame, syncs memory
            llm,  # LLM
            assistant_response_logger,  # Captures assistant responses and logs to JSON
            tts,  # TTS
            transport.output(),  # Transport bot output
            audio_buffer,  # Audio recording - captures both user and bot audio
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
        # Start audio recording
        await audio_buffer.start_recording()
        logger.info(f"Audio recording started. Session: {conversation_logger.session_id}")
        # Kick off the conversation.
        # Note: We don't add to messages list anymore since MemoryContextManager handles context
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        # End the conversation span for V2V tracing
        v2v_tracer.end_conversation()
        
        # Stop audio recording
        await audio_buffer.stop_recording()
        
        # Log recording summary
        logger.info(
            f"Recording complete. Session: {conversation_logger.session_id}, "
            f"User turns: {conversation_logger.user_turn_count}, "
            f"Bot turns: {conversation_logger.bot_turn_count}, "
            f"Audio user turns: {user_audio_turn}, "
            f"Audio bot turns: {bot_audio_turn}"
        )
        logger.info(f"Conversation JSON saved to: {conversation_logger.json_file}")
        logger.info(f"Audio files saved to: {conversation_logger.session_dir}")
        
        # Log memory state on disconnect
        state = conversation_memory.get_state()
        logger.info(
            f"Conversation memory state: "
            f"Total messages: {state.total_messages_processed}, "
            f"Summarized: {state.messages_summarized}, "
            f"Recent: {len(state.recent_messages)}, "
            f"Has summary: {conversation_memory.has_summary}"
        )
        if conversation_memory.has_summary:
            logger.info(f"Final summary: {state.summary[:200]}...")
        
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