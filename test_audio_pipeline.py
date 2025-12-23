#
# Audio Pipeline Test Script
#
# Sends audio files sequentially to the bot pipeline, waiting for bot response
# before sending the next audio file.
#
# This test uses the ACTUAL bot.py components (MemoryContextManager, 
# AssistantResponseLogger, ConversationLogger, etc.) to test the real bot logic.
#

import os
import asyncio
import json
import wave
from pathlib import Path
from typing import Optional, List, Callable

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    EndFrame,
    TextFrame,
    TranscriptionFrame,
    LLMContextFrame,
    LLMRunFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transcriptions.language import Language

# Import actual bot.py components for testing the real bot logic
from bot import (
    MemoryContextManager,
    AssistantResponseLogger,
    ConversationLogger,
    load_system_prompt,
    load_voice_cloning_key,
    fetch_user_memories,
    USER_NAME,
)
from async_memory import AsyncConversationMemory
from tracer import V2VSpeakingTracer

load_dotenv(override=True)

# Setup OpenTelemetry tracing for V2V events (same as bot.py)
V2V_TRACER = None
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # Setup tracer for v2v speaking events with service name "pipecat_v2v_test"
    v2v_resource = Resource.create({
        "service.name": "pipecat_v2v_test",
        "service.instance.id": os.getenv("HOSTNAME", "test"),
        "deployment.environment": "testing",
    })
    v2v_tracer_provider = TracerProvider(resource=v2v_resource)
    v2v_exporter = OTLPSpanExporter(
        endpoint="http://localhost:4317",
        insecure=True,
    )
    v2v_tracer_provider.add_span_processor(BatchSpanProcessor(v2v_exporter))
    V2V_TRACER = v2v_tracer_provider.get_tracer("pipecat_v2v_test")
    logger.info("V2V OpenTelemetry tracer initialized for testing with service name 'pipecat_v2v_test'")

except ImportError as e:
    logger.warning(f"Failed to import tracing dependencies: {e}. V2V tracing will be disabled.")
except Exception as e:
    logger.warning(f"Failed to setup V2V tracing: {e}. V2V tracing will be disabled.")


class AudioFileSource(FrameProcessor):
    """
    A FrameProcessor that reads audio files and injects them into the pipeline.
    
    This replaces the transport.input() for testing purposes.
    """
    
    def __init__(self, sample_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self._sample_rate = sample_rate
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._started = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Pass through frames from downstream."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
    
    async def queue_audio_file(self, audio_path: Path):
        """Queue an audio file to be sent."""
        await self._audio_queue.put(audio_path)
    
    async def send_audio_file(self, audio_path: Path, chunk_duration_ms: int = 20):
        """
        Send an audio file through the pipeline.
        
        Args:
            audio_path: Path to the WAV file
            chunk_duration_ms: Duration of each audio chunk in milliseconds
        """
        logger.info(f"AudioFileSource: Loading audio file: {audio_path}")
        
        # Load WAV file
        with wave.open(str(audio_path), 'rb') as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            audio_data = wf.readframes(wf.getnframes())
        
        logger.info(
            f"AudioFileSource: Loaded {len(audio_data)} bytes, "
            f"sample_rate={sample_rate}, channels={num_channels}, "
            f"sample_width={sample_width}"
        )
        
        # Calculate chunk size (in bytes)
        # chunk_duration_ms / 1000 * sample_rate * sample_width * num_channels
        chunk_size = int(chunk_duration_ms / 1000 * sample_rate * sample_width * num_channels)
        
        # Send UserStartedSpeakingFrame
        logger.info("AudioFileSource: Sending UserStartedSpeakingFrame")
        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        
        # Send audio in chunks (simulate streaming)
        chunk_count = 0
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            frame = InputAudioRawFrame(
                audio=chunk,
                sample_rate=sample_rate,
                num_channels=num_channels,
            )
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)
            chunk_count += 1
            
            # Simulate real-time streaming delay
            await asyncio.sleep(chunk_duration_ms / 1000)
        
        logger.info(f"AudioFileSource: Sent {chunk_count} audio chunks")
        
        # Send UserStoppedSpeakingFrame
        logger.info("AudioFileSource: Sending UserStoppedSpeakingFrame")
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)


class BotResponseMonitor(FrameProcessor):
    """
    A FrameProcessor that monitors bot responses and signals when complete.
    
    This allows the test harness to know when the bot has finished speaking.
    It tracks both BotStoppedSpeakingFrame and TTSStoppedFrame for compatibility.
    """
    
    def __init__(self, tts_buffer_time: float = 2.0, **kwargs):
        """
        Initialize the monitor.
        
        Args:
            tts_buffer_time: Extra time to wait after LLM response for TTS to complete
        """
        super().__init__(**kwargs)
        self._bot_speaking = False
        self._response_complete_event = asyncio.Event()
        self._llm_response_complete_event = asyncio.Event()
        self._tts_complete_event = asyncio.Event()
        self._on_bot_response_complete: Optional[Callable] = None
        self._current_response_text: List[str] = []
        self._tts_buffer_time = tts_buffer_time
        self._tts_active = False
        self._pending_tts_complete_task: Optional[asyncio.Task] = None
        self._pending_delayed_response_task: Optional[asyncio.Task] = None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Monitor frames for bot response status."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._response_complete_event.clear()
            logger.info("BotResponseMonitor: Bot started speaking")
        
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._response_complete_event.set()
            logger.info("BotResponseMonitor: Bot stopped speaking")
            
            if self._on_bot_response_complete:
                await self._on_bot_response_complete()
        
        elif isinstance(frame, TTSStartedFrame):
            self._tts_active = True
            self._tts_complete_event.clear()
            # Cancel any pending completion task
            if self._pending_tts_complete_task:
                self._pending_tts_complete_task.cancel()
                self._pending_tts_complete_task = None
            logger.debug("BotResponseMonitor: TTS started")
        
        elif isinstance(frame, TTSStoppedFrame):
            self._tts_active = False
            # Schedule completion after a small buffer (in case multiple TTS chunks)
            self._pending_tts_complete_task = asyncio.create_task(
                self._delayed_tts_complete()
            )
            logger.debug("BotResponseMonitor: TTS stopped, scheduling completion check")
        
        elif isinstance(frame, TextFrame):
            self._current_response_text.append(frame.text)
        
        elif isinstance(frame, LLMFullResponseEndFrame):
            full_response = "".join(self._current_response_text)
            logger.info(f"BotResponseMonitor: LLM response complete: {full_response[:100]}...")
            self._current_response_text = []
            self._llm_response_complete_event.set()
            
            # Cancel any existing delayed response task before creating a new one
            if self._pending_delayed_response_task:
                self._pending_delayed_response_task.cancel()
            
            # Schedule response completion after buffer time if no TTS frames
            self._pending_delayed_response_task = asyncio.create_task(
                self._delayed_response_complete()
            )
        
        await self.push_frame(frame, direction)
    
    async def _delayed_tts_complete(self):
        """Mark TTS as complete after a short delay (handles multiple chunks)."""
        await asyncio.sleep(0.5)
        if not self._tts_active:
            self._tts_complete_event.set()
            logger.debug("BotResponseMonitor: TTS complete event set")
    
    async def _delayed_response_complete(self):
        """Mark response as complete after buffer time for TTS."""
        await asyncio.sleep(self._tts_buffer_time)
        if not self._response_complete_event.is_set():
            self._response_complete_event.set()
            logger.info("BotResponseMonitor: Response complete (via LLM + TTS buffer)")
    
    async def wait_for_response(self, timeout: float = 60.0) -> bool:
        """
        Wait for the bot to complete its response.
        
        This waits for either:
        - BotStoppedSpeakingFrame (from transport)
        - Or LLMFullResponseEndFrame + TTS buffer time
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if response completed, False if timeout
        """
        # Clear event and cancel any pending completion tasks BEFORE waiting
        # This ensures we wait for the NEXT response, not a stale one
        self._response_complete_event.clear()
        self._llm_response_complete_event.clear()
        self._tts_complete_event.clear()
        
        # Cancel any pending delayed completion tasks
        if self._pending_tts_complete_task:
            self._pending_tts_complete_task.cancel()
            self._pending_tts_complete_task = None
        if self._pending_delayed_response_task:
            self._pending_delayed_response_task.cancel()
            self._pending_delayed_response_task = None
        
        try:
            await asyncio.wait_for(self._response_complete_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"BotResponseMonitor: Timeout waiting for bot response")
            return False
    
    async def wait_for_llm_response(self, timeout: float = 30.0) -> bool:
        """
        Wait for the LLM to complete its response (before TTS).
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if response completed, False if timeout
        """
        try:
            await asyncio.wait_for(self._llm_response_complete_event.wait(), timeout)
            self._llm_response_complete_event.clear()
            return True
        except asyncio.TimeoutError:
            logger.warning(f"BotResponseMonitor: Timeout waiting for LLM response")
            return False
    
    def set_on_response_complete(self, callback: Callable):
        """Set a callback to be called when bot finishes speaking."""
        self._on_bot_response_complete = callback
    
    @property
    def is_speaking(self) -> bool:
        """Check if bot is currently speaking."""
        return self._bot_speaking


class NullAudioSink(FrameProcessor):
    """
    A FrameProcessor that consumes audio output (replaces transport.output()).
    
    For testing, we don't need to actually play the audio, just consume it.
    This also generates BotStartedSpeakingFrame and BotStoppedSpeakingFrame
    to simulate what a real transport would do.
    
    The speaking frames are pushed UPSTREAM so they reach the V2VSpeakingTracer
    which is earlier in the pipeline.
    """
    
    def __init__(self, save_audio: bool = False, output_dir: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self._save_audio = save_audio
        self._output_dir = output_dir or Path("test_output")
        self._audio_buffer: List[bytes] = []
        self._response_count = 0
        self._bot_speaking = False
        self._tts_active = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Consume audio frames and generate speaking frames."""
        await super().process_frame(frame, direction)
        
        # Track TTS start/stop to generate speaking frames
        # Push them UPSTREAM so V2VSpeakingTracer (earlier in pipeline) can see them
        if isinstance(frame, TTSStartedFrame):
            self._tts_active = True
            if not self._bot_speaking:
                self._bot_speaking = True
                logger.info("NullAudioSink: Generating BotStartedSpeakingFrame (upstream)")
                # Push upstream so V2VSpeakingTracer receives it
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
                # Also push downstream for BotResponseMonitor
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        
        elif isinstance(frame, TTSStoppedFrame):
            self._tts_active = False
            if self._bot_speaking:
                self._bot_speaking = False
                logger.info("NullAudioSink: Generating BotStoppedSpeakingFrame (upstream)")
                # Push upstream so V2VSpeakingTracer receives it
                await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
                # Also push downstream for BotResponseMonitor
                await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        
        # Pass everything through downstream
        await self.push_frame(frame, direction)


class UserMessageLogger(FrameProcessor):
    """
    Logs user messages from LLMContextFrame.
    
    Should be placed BEFORE the LLM in the pipeline (after context_aggregator.user()).
    """
    
    def __init__(self, conversation_logger: ConversationLogger, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._conversation_logger = conversation_logger
        self._context = context
        self._last_user_message: Optional[str] = None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Capture user messages from LLMContextFrame (created by context_aggregator.user())
        if isinstance(frame, LLMContextFrame):
            self._capture_user_from_context()
            self._log_llm_context()
        
        await self.push_frame(frame, direction)
    
    def _capture_user_from_context(self):
        """Extract and log the latest user message from LLM context."""
        messages = self._context.get_messages()
        
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_content = msg.get("content", "")
                if user_content and user_content != self._last_user_message:
                    self._last_user_message = user_content
                    self._conversation_logger.add_message("user", user_content)
                    logger.debug(f"UserMessageLogger: Logged user message: {user_content[:50]}...")
                break
    
    def _log_llm_context(self):
        """Log all messages being sent to LLM for debugging."""
        messages = self._context.get_messages()
        logger.info(f"UserMessageLogger: LLM Context ({len(messages)} messages):")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            content_preview = content[:100] + "..." if len(content) > 100 else content
            logger.debug(f"  [{i}] {role}: {content_preview}")


class AssistantMessageLogger(FrameProcessor):
    """
    Logs assistant messages from TextFrame/LLMFullResponseEndFrame.
    
    Should be placed AFTER the LLM in the pipeline.
    """
    
    def __init__(self, conversation_logger: ConversationLogger, **kwargs):
        super().__init__(**kwargs)
        self._conversation_logger = conversation_logger
        self._llm_response_started = False
        self._assistant_aggregation: List[str] = []
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMFullResponseStartFrame):
            self._llm_response_started = True
            self._assistant_aggregation = []
        
        elif isinstance(frame, TextFrame) and self._llm_response_started:
            if frame.text:
                self._assistant_aggregation.append(frame.text)
        
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_response_started = False
            
            if self._assistant_aggregation:
                full_response = "".join(self._assistant_aggregation)
                self._assistant_aggregation = []
                
                if full_response.strip():
                    self._conversation_logger.add_message("assistant", full_response)
                    logger.debug(f"AssistantMessageLogger: Logged assistant message: {full_response[:50]}...")
        
        await self.push_frame(frame, direction)


class AudioPipelineTester:
    """
    Main test harness for sending audio files to the bot pipeline.
    
    This uses the ACTUAL bot.py components to test the real bot logic:
    - MemoryContextManager for conversation memory
    - AssistantResponseLogger for response logging  
    - ConversationLogger for JSON/audio logging
    - V2VSpeakingTracer for OpenTelemetry tracing
    
    Usage:
        tester = AudioPipelineTester()
        await tester.initialize()
        await tester.send_audio_files_from_directory("recordings/session_1")
        await tester.shutdown()
    """
    
    def __init__(
        self,
        user_name: Optional[str] = None,
        sample_rate: int = 16000,
        use_bot_components: bool = True,
        with_memory: bool = False,
        conversation_history_path: Optional[str] = None,
    ):
        """
        Initialize the tester.
        
        Args:
            user_name: User name for system prompt interpolation (default: from bot.py)
            sample_rate: Audio sample rate
            use_bot_components: If True, use actual bot.py components. If False, use simplified pipeline.
            with_memory: If True, include MemoryContextManager and fetch user memories. Default: False.
            conversation_history_path: Path to a conversation.json file to pre-populate the LLM context.
        """
        self._user_name = user_name or USER_NAME
        self._sample_rate = sample_rate
        self._use_bot_components = use_bot_components
        self._with_memory = with_memory
        self._conversation_history_path = conversation_history_path
        
        # Pipeline components
        self._audio_source: Optional[AudioFileSource] = None
        self._response_monitor: Optional[BotResponseMonitor] = None
        self._pipeline: Optional[Pipeline] = None
        self._task: Optional[PipelineTask] = None
        self._runner: Optional[PipelineRunner] = None
        self._pipeline_task: Optional[asyncio.Task] = None
        
        # Bot components (from bot.py)
        self._conversation_memory: Optional[AsyncConversationMemory] = None
        self._conversation_logger: Optional[ConversationLogger] = None
        self._memory_context_manager: Optional[MemoryContextManager] = None
        self._assistant_response_logger: Optional[AssistantResponseLogger] = None
        self._user_message_logger: Optional[UserMessageLogger] = None
        self._assistant_message_logger: Optional[AssistantMessageLogger] = None
        self._v2v_tracer: Optional[V2VSpeakingTracer] = None
        
        # State
        self._initialized = False
        self._pipeline_ready = asyncio.Event()
        self._system_prompt: Optional[str] = None
    
    def _load_system_prompt(self) -> str:
        """Load and interpolate system prompt from bot.py."""
        # Only fetch user memories if with_memory is enabled
        if self._with_memory:
            user_memories = fetch_user_memories(self._user_name)
            logger.info(f"AudioPipelineTester: Fetched user memories (with_memory=True)")
        else:
            user_memories = ""
            logger.info(f"AudioPipelineTester: Skipping user memories (with_memory=False)")
        
        system_prompt = load_system_prompt()
        return system_prompt.replace("{user_name}", self._user_name).replace("{user_memories}", user_memories)
    
    def _load_conversation_history(self) -> List[dict]:
        """
        Load conversation history from a JSON file.
        
        Returns:
            List of message dicts with 'role' and 'content' keys for LLM context.
        """
        if not self._conversation_history_path:
            return []
        
        history_path = Path(self._conversation_history_path)
        if not history_path.exists():
            logger.warning(f"AudioPipelineTester: Conversation history file not found: {history_path}")
            return []
        
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            messages = data.get("messages", [])
            
            # Convert to LLM context format (only role and content)
            context_messages = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                
                # Only include user and assistant messages
                if role in ("user", "assistant") and content:
                    context_messages.append({
                        "role": role,
                        "content": content,
                    })
            
            logger.info(
                f"AudioPipelineTester: Loaded {len(context_messages)} messages from conversation history "
                f"({history_path.name})"
            )
            return context_messages
            
        except json.JSONDecodeError as e:
            logger.error(f"AudioPipelineTester: Failed to parse conversation history: {e}")
            return []
        except Exception as e:
            logger.error(f"AudioPipelineTester: Error loading conversation history: {e}")
            return []
    
    async def initialize(self):
        """Initialize the pipeline using actual bot.py components."""
        memory_mode = "with memory" if self._with_memory else "without memory"
        logger.info(f"AudioPipelineTester: Initializing pipeline ({memory_mode})...")
        
        # Load system prompt
        self._system_prompt = self._load_system_prompt()
        logger.info(f"AudioPipelineTester: Loaded system prompt for user: {self._user_name}")
        
        # Create STT service (same as bot.py)
        stt = OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-transcribe",
            language=Language.HI_IN,
        )
        
        # Create TTS service (same as bot.py)
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        credentials_content = None
        if credentials_path:
            try:
                with open(credentials_path, "r", encoding="utf-8") as f:
                    credentials_content = f.read()
            except FileNotFoundError:
                logger.warning(f"Google credentials file not found: {credentials_path}")
        
        voice_cloning_key = load_voice_cloning_key()
        tts_kwargs = {
            "voice_id": "en-US-Chirp3-HD-Charon",
            "params": GoogleTTSService.InputParams(language=Language.EN_US),
        }
        if credentials_content:
            tts_kwargs["credentials"] = credentials_content
        if voice_cloning_key:
            tts_kwargs["voice_cloning_key"] = voice_cloning_key
        
        tts = GoogleTTSService(**tts_kwargs)
        
        # Create LLM service
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create LLMContext with system prompt and optional conversation history
        messages = [
            {"role": "system", "content": self._system_prompt},
        ]
        
        # Load and add conversation history if provided
        conversation_history = self._load_conversation_history()
        if conversation_history:
            messages.extend(conversation_history)
            logger.info(
                f"AudioPipelineTester: Pre-populated context with {len(conversation_history)} messages "
                f"from conversation history"
            )
        
        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)
        
        # Create ConversationLogger (from bot.py)
        self._conversation_logger = ConversationLogger()
        logger.info(f"AudioPipelineTester: Created ConversationLogger. Session: {self._conversation_logger.session_id}")
        
        # Only create memory components if with_memory is enabled
        if self._with_memory:
            # Initialize AsyncConversationMemory (same as bot.py)
            self._conversation_memory = AsyncConversationMemory(
                api_key=os.getenv("OPENAI_API_KEY"),
                token_limit=200,
                model="gpt-4o-mini",
                auto_summarize=True,
                summarize_prompt="""Summarize the conversation concisely. 
Preserve: names, numbers, key decisions, action items, and important context.
Keep it brief but comprehensive enough to continue the conversation naturally."""
            )
            
            # Create MemoryContextManager (from bot.py)
            self._memory_context_manager = MemoryContextManager(
                memory=self._conversation_memory,
                context=context,
                system_prompt=self._system_prompt,
                conversation_logger=self._conversation_logger,
            )
            
            # Create AssistantResponseLogger (from bot.py)
            self._assistant_response_logger = AssistantResponseLogger(
                memory=self._conversation_memory,
                conversation_logger=self._conversation_logger,
            )
            logger.info("AudioPipelineTester: Memory components enabled")
        else:
            logger.info("AudioPipelineTester: Memory components disabled (with_memory=False)")
        
        # Create V2VSpeakingTracer (from bot.py) - with OpenTelemetry for testing
        self._v2v_tracer = V2VSpeakingTracer(tracer=V2V_TRACER)
        
        # Create our custom processors
        self._audio_source = AudioFileSource(sample_rate=self._sample_rate)
        self._response_monitor = BotResponseMonitor()
        audio_sink = NullAudioSink()
        
        # Build pipeline - structure depends on with_memory flag
        if self._with_memory:
            # Full pipeline with memory management (same as bot.py)
            self._pipeline = Pipeline(
                [
                    self._audio_source,  # Inject audio files (replaces transport.input())
                    stt,  # STT
                    self._v2v_tracer,  # V2V speaking event tracer
                    context_aggregator.user(),  # User responses - creates LLMContextFrame
                    self._memory_context_manager,  # Intercepts LLMContextFrame, syncs memory
                    llm,  # LLM
                    self._assistant_response_logger,  # Captures assistant responses
                    tts,  # TTS
                    self._response_monitor,  # Monitor bot responses for test coordination
                    audio_sink,  # Consume audio output (replaces transport.output())
                    context_aggregator.assistant(),  # Assistant spoken responses
                ]
            )
            logger.info("AudioPipelineTester: Built pipeline WITH memory components")
        else:
            # Create loggers for conversation logging without memory management
            self._user_message_logger = UserMessageLogger(
                conversation_logger=self._conversation_logger,
                context=context,
            )
            self._assistant_message_logger = AssistantMessageLogger(
                conversation_logger=self._conversation_logger,
            )
            
            # Simplified pipeline without memory management but with conversation logging
            # UserMessageLogger is placed BEFORE LLM to capture user messages from LLMContextFrame
            # AssistantMessageLogger is placed AFTER LLM to capture assistant responses
            self._pipeline = Pipeline(
                [
                    self._audio_source,  # Inject audio files (replaces transport.input())
                    stt,  # STT
                    self._v2v_tracer,  # V2V speaking event tracer
                    context_aggregator.user(),  # User responses - creates LLMContextFrame
                    self._user_message_logger,  # Log user messages from LLMContextFrame
                    llm,  # LLM
                    self._assistant_message_logger,  # Log assistant responses
                    tts,  # TTS
                    self._response_monitor,  # Monitor bot responses for test coordination
                    audio_sink,  # Consume audio output (replaces transport.output())
                    context_aggregator.assistant(),  # Assistant spoken responses
                ]
            )
            logger.info("AudioPipelineTester: Built pipeline WITHOUT memory components (with conversation logging)")
        
        self._task = PipelineTask(
            self._pipeline,
            params=PipelineParams(
                audio_out_sample_rate=24000,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        
        self._runner = PipelineRunner(handle_sigint=False)
        
        self._initialized = True
        logger.info(f"AudioPipelineTester: Pipeline initialized ({memory_mode})")
        
        # Start pipeline in background
        self._pipeline_task = asyncio.create_task(self._run_pipeline_internal())
        
        # Wait for pipeline to be ready
        await asyncio.sleep(0.5)
        self._pipeline_ready.set()
        
        # Start conversation tracing
        self._v2v_tracer.start_conversation()
        
        logger.info("AudioPipelineTester: Pipeline started and ready")
    
    async def _run_pipeline_internal(self):
        """Internal method to run the pipeline."""
        try:
            await self._runner.run(self._task)
        except asyncio.CancelledError:
            logger.info("AudioPipelineTester: Pipeline task cancelled")
        except Exception as e:
            logger.error(f"AudioPipelineTester: Pipeline error: {e}")
            import traceback
            traceback.print_exc()
    
    async def send_audio_file(
        self,
        audio_path: Path,
        wait_for_response: bool = True,
        response_timeout: float = 60.0,
    ) -> bool:
        """
        Send a single audio file and optionally wait for bot response.
        
        Args:
            audio_path: Path to the WAV file
            wait_for_response: If True, wait for bot to finish speaking
            response_timeout: Maximum time to wait for response
            
        Returns:
            True if successful, False if error or timeout
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        logger.info(f"AudioPipelineTester: Sending audio file: {audio_path}")
        
        # Send the audio file
        await self._audio_source.send_audio_file(audio_path)
        
        # Wait for bot response if requested
        if wait_for_response:
            logger.info("AudioPipelineTester: Waiting for bot response...")
            success = await self._response_monitor.wait_for_response(timeout=response_timeout)
            if success:
                logger.info("AudioPipelineTester: Bot response complete")
            else:
                logger.warning("AudioPipelineTester: Timeout waiting for bot response")
            return success
        
        return True
    
    async def trigger_initial_greeting(
        self,
        wait_for_response: bool = True,
        response_timeout: float = 60.0,
    ) -> bool:
        """
        Trigger the bot's initial greeting by sending LLMRunFrame.
        
        This mimics what happens in bot.py when a client connects.
        
        Args:
            wait_for_response: If True, wait for bot to finish speaking
            response_timeout: Maximum time to wait for response
            
        Returns:
            True if successful, False if error or timeout
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        logger.info("AudioPipelineTester: Triggering initial greeting (LLMRunFrame)")
        
        # Queue LLMRunFrame to trigger the LLM to generate greeting
        await self._task.queue_frames([LLMRunFrame()])
        
        # Wait for bot response if requested
        if wait_for_response:
            logger.info("AudioPipelineTester: Waiting for initial greeting...")
            success = await self._response_monitor.wait_for_response(timeout=response_timeout)
            if success:
                logger.info("AudioPipelineTester: Initial greeting complete")
            else:
                logger.warning("AudioPipelineTester: Timeout waiting for initial greeting")
            return success
        
        return True
    
    async def send_audio_files(
        self,
        audio_paths: List[Path],
        wait_for_response: bool = True,
        response_timeout: float = 60.0,
        delay_between_files: float = 0.5,
        trigger_greeting: bool = True,
    ) -> List[bool]:
        """
        Send multiple audio files sequentially, waiting for bot response between each.
        
        Args:
            audio_paths: List of paths to WAV files
            wait_for_response: If True, wait for bot to finish speaking after each file
            response_timeout: Maximum time to wait for each response
            delay_between_files: Delay between files in seconds
            trigger_greeting: If True, trigger initial bot greeting before first audio
            
        Returns:
            List of success status for each file
        """
        results = []
        
        # Trigger initial greeting if requested
        if trigger_greeting:
            greeting_success = await self.trigger_initial_greeting(
                wait_for_response=wait_for_response,
                response_timeout=response_timeout,
            )
            if not greeting_success:
                logger.warning("AudioPipelineTester: Initial greeting failed or timed out")
            
            # Add delay after greeting
            if delay_between_files > 0:
                await asyncio.sleep(delay_between_files)
        
        for i, audio_path in enumerate(audio_paths):
            logger.info(f"AudioPipelineTester: Processing file {i+1}/{len(audio_paths)}: {audio_path.name}")
            
            success = await self.send_audio_file(
                audio_path,
                wait_for_response=wait_for_response,
                response_timeout=response_timeout,
            )
            results.append(success)
            
            # Add delay between files
            if i < len(audio_paths) - 1 and delay_between_files > 0:
                await asyncio.sleep(delay_between_files)
        
        return results
    
    async def send_audio_files_from_directory(
        self,
        directory_path: str,
        pattern: str = "user_turn_*.wav",
        wait_for_response: bool = True,
        response_timeout: float = 60.0,
        delay_between_files: float = 0.5,
        trigger_greeting: bool = True,
    ) -> List[bool]:
        """
        Send all matching audio files from a directory sequentially.
        
        Args:
            directory_path: Path to the directory containing audio files
            pattern: Glob pattern to match audio files
            wait_for_response: If True, wait for bot to finish speaking after each file
            response_timeout: Maximum time to wait for each response
            delay_between_files: Delay between files in seconds
            trigger_greeting: If True, trigger initial bot greeting before first audio
            
        Returns:
            List of success status for each file
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all matching audio files, sorted numerically by the number in the filename
        def extract_number(path: Path) -> int:
            """Extract the numeric part from filename like 'user_turn_1.wav' -> 1"""
            import re
            match = re.search(r'(\d+)', path.stem)
            return int(match.group(1)) if match else 0
        
        audio_files = sorted(directory.glob(pattern), key=extract_number)
        
        if not audio_files:
            logger.warning(f"No audio files found matching pattern '{pattern}' in {directory}")
            return []
        
        logger.info(f"AudioPipelineTester: Found {len(audio_files)} audio files in {directory}")
        for f in audio_files:
            logger.info(f"  - {f.name}")
        
        return await self.send_audio_files(
            audio_files,
            wait_for_response=wait_for_response,
            response_timeout=response_timeout,
            delay_between_files=delay_between_files,
            trigger_greeting=trigger_greeting,
        )
    
    async def shutdown(self):
        """Shutdown the pipeline gracefully."""
        logger.info("AudioPipelineTester: Shutting down...")
        
        # End conversation tracing
        if self._v2v_tracer:
            self._v2v_tracer.end_conversation()
        
        # Log memory state
        if self._conversation_memory:
            state = self._conversation_memory.get_state()
            logger.info(
                f"AudioPipelineTester: Memory state - "
                f"Total messages: {state.total_messages_processed}, "
                f"Summarized: {state.messages_summarized}, "
                f"Recent: {len(state.recent_messages)}, "
                f"Has summary: {self._conversation_memory.has_summary}"
            )
            if self._conversation_memory.has_summary:
                logger.info(f"AudioPipelineTester: Final summary: {state.summary[:200]}...")
        
        # Log conversation logger stats
        if self._conversation_logger:
            logger.info(
                f"AudioPipelineTester: Conversation logged - "
                f"User turns: {self._conversation_logger.user_turn_count}, "
                f"Bot turns: {self._conversation_logger.bot_turn_count}"
            )
            logger.info(f"AudioPipelineTester: Conversation JSON saved to: {self._conversation_logger.json_file}")
        
        if self._task and self._audio_source:
            # Send end frame
            await self._audio_source.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
            await self._task.cancel()
        
        if self._pipeline_task:
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass
        
        self._initialized = False
        logger.info("AudioPipelineTester: Shutdown complete")
    
    @property
    def conversation_logger(self) -> Optional[ConversationLogger]:
        """Get the conversation logger."""
        return self._conversation_logger
    
    @property
    def conversation_memory(self) -> Optional[AsyncConversationMemory]:
        """Get the conversation memory."""
        return self._conversation_memory


async def run_test(
    audio_directory: str,
    pattern: str = "user_turn_*.wav",
    wait_for_response: bool = True,
    with_memory: bool = False,
    conversation_history_path: Optional[str] = None,
):
    """
    Run a test sending audio files from a directory.
    
    Args:
        audio_directory: Path to directory containing audio files
        pattern: Glob pattern to match audio files
        wait_for_response: If True, wait for bot response after each file
        with_memory: If True, enable MemoryContextManager and user memories
        conversation_history_path: Path to a conversation.json to pre-populate context
    """
    tester = AudioPipelineTester(
        with_memory=with_memory,
        conversation_history_path=conversation_history_path,
    )
    
    try:
        await tester.initialize()
        
        # Send audio files
        results = await tester.send_audio_files_from_directory(
            audio_directory,
            pattern=pattern,
            wait_for_response=wait_for_response,
            response_timeout=60.0,
            delay_between_files=1.0,
        )
        
        # Log results
        success_count = sum(results) if results else 0
        total_count = len(results) if results else 0
        logger.info(f"Test complete. Results: {success_count}/{total_count} successful")
        
        # Cleanup
        await tester.shutdown()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


async def main():
    """Main entry point for the test script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test bot pipeline with audio files")
    parser.add_argument(
        "audio_directory",
        type=str,
        help="Path to directory containing audio files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="user_turn_*.wav",
        help="Glob pattern for audio files (default: user_turn_*.wav)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for bot response (fire and forget)",
    )
    parser.add_argument(
        "--with-memory",
        action="store_true",
        default=False,
        help="Enable MemoryContextManager and fetch user memories (default: False)",
    )
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to conversation.json to pre-populate LLM context with conversation history",
    )
    
    args = parser.parse_args()
    
    await run_test(
        audio_directory=args.audio_directory,
        pattern=args.pattern,
        wait_for_response=not args.no_wait,
        with_memory=args.with_memory,
        conversation_history_path=args.history,
    )


if __name__ == "__main__":
    asyncio.run(main())

