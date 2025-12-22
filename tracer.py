#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


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
        # Track bot speaking state and interruptions
        self._bot_is_speaking = False
        self._is_interruption_turn = False  # Current user turn is an interruption
        self._current_transcription = ""  # Accumulate transcription text

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
            elif isinstance(frame, TranscriptionFrame):
                self._handle_transcription(frame)

        # Always pass the frame through
        await self.push_frame(frame, direction)

    def _handle_user_started_speaking(self):
        """Create a span when user starts speaking (child of conversation)."""
        # Check if this is an interruption (user started speaking while bot was speaking)
        if self._bot_is_speaking:
            self._is_interruption_turn = True
            logger.info("V2V Tracer: Interruption detected - user started speaking while bot was speaking")
        else:
            self._is_interruption_turn = False
        
        # Reset transcription accumulator for new turn
        self._current_transcription = ""
        
        if self._user_speaking_span is None and self._conversation_context is not None:
            self._user_speaking_start_time = time.time()
            self._user_speaking_span = self._tracer.start_span(
                "user_speaking",
                context=self._conversation_context,
                attributes={
                    "event.type": "user_speaking",
                    "speaking.start_time": self._user_speaking_start_time,
                    "interrupted_speech": self._is_interruption_turn,  # Set immediately
                }
            )
            logger.info(f"V2V Tracer: User started speaking span created (interrupted_speech={self._is_interruption_turn})")

    def _handle_user_stopped_speaking(self):
        """End the span when user stops speaking."""
        if self._user_speaking_span is not None:
            duration = time.time() - self._user_speaking_start_time if self._user_speaking_start_time else 0
            self._user_speaking_span.set_attribute("speaking.duration_seconds", duration)
            self._user_speaking_span.set_attribute("speaking.end_time", time.time())
            # Ensure final transcription is set
            if self._current_transcription:
                self._user_speaking_span.set_attribute("speech.text", self._current_transcription)
            self._user_speaking_span.end()
            logger.info(f"V2V Tracer: User stopped speaking (duration: {duration:.2f}s, interrupted_speech={self._is_interruption_turn}, text='{self._current_transcription}')")
            self._user_speaking_span = None
            self._user_speaking_start_time = None
            self._current_transcription = ""

    def _handle_bot_started_speaking(self):
        """Create a span when bot starts speaking (child of conversation)."""
        self._bot_is_speaking = True
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
        self._bot_is_speaking = False
        if self._bot_speaking_span is not None:
            duration = time.time() - self._bot_speaking_start_time if self._bot_speaking_start_time else 0
            self._bot_speaking_span.set_attribute("speaking.duration_seconds", duration)
            self._bot_speaking_span.set_attribute("speaking.end_time", time.time())
            self._bot_speaking_span.end()
            self._bot_speaking_span = None
            self._bot_speaking_start_time = None
            logger.debug(f"V2V Tracer: Bot stopped speaking span ended (duration: {duration:.2f}s)")

    def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle transcription frame - add transcription text to the span."""
        if frame.text:
            # Accumulate transcription text
            if self._current_transcription:
                self._current_transcription += " " + frame.text
            else:
                self._current_transcription = frame.text
            
            # Update the span with the transcription text
            if self._user_speaking_span is not None:
                self._user_speaking_span.set_attribute("speech.text", self._current_transcription)
                logger.info(f"V2V Tracer: Added transcription to span: '{frame.text}' (interrupted_speech={self._is_interruption_turn})")

