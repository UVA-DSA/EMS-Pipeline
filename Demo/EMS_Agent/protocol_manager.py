"""
protocol_manager.py

Wraps EMSAgentInference (from EMS_Agent/Interface/EMSTinyBERTSystem.py) into a
manager with the same start/stop interface as SpeechProcessManager.

Usage in GUI.py:
    self.ProtocolManager = ProtocolManager()
    self.ProtocolManager.protocol_prediction.connect(self.UpdateProtocolBoxes)
    self.ProtocolManager.start()
    ...
    # Feed a transcript line to the model:
    self.ProtocolManager.feed_transcript(transcript_text)
    ...
    self.ProtocolManager.stop()
"""

import queue
import time
from PyQt5.QtCore import QThread, pyqtSignal


# ── Lightweight transcript carrier ───────────────────────────────────────────
# EMSAgentInference expects queue items with a .transcript attribute.
# We also add stub fields that the evaluation branch reads so it doesn't crash
# even when pipeline_config.evaluation is False.

class TranscriptItem:
    """Minimal object that satisfies EMSAgentInference's queue item interface."""
    def __init__(self, transcript: str):
        self.transcript = transcript
        self.transcriptionDuration = 0
        self.confidence = 0.0


# ── Lightweight protocol result carrier ──────────────────────────────────────
# UpdateProtocolBoxes reads .protocol and .protocol_confidence from each item
# in the list it receives.

class ProtocolPrediction:
    """Wraps a single (protocol, confidence) pair for the GUI."""
    def __init__(self, protocol: str, confidence: float):
        self.protocol = protocol
        self.protocol_confidence = confidence
        self.intervention = ""   # stub — InterventionBox reads this


# ── Manager ──────────────────────────────────────────────────────────────────

class ProtocolManager:
    """
    Loads the EMSTinyBERT protocol model in a background QThread and feeds it
    transcripts via a queue.  Emits protocol_prediction(list[ProtocolPrediction])
    for the GUI to display.

    The model is heavy (~10 s load), so initialisation happens inside the
    thread's run() so the GUI stays responsive.
    """

    def __init__(self):
        self._speech_queue: queue.Queue = queue.Queue(maxsize=100)
        self._thread = _ProtocolThread(self._speech_queue)
        print('[ProtocolManager] Initialized')

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        print('[ProtocolManager] Starting...')
        self._thread.start()
        print('[ProtocolManager] Started')

    def stop(self):
        print('[ProtocolManager] Stopping...')
        self._thread.stop()
        # Unblock the queue.get() inside run()
        try:
            self._speech_queue.put_nowait('Kill')
        except queue.Full:
            pass
        print('[ProtocolManager] Stopped')

    def feed_transcript(self, transcript: str):
        """
        Call this every time a new transcript line arrives (from Whisper or
        Google STT).  Non-blocking — drops the item if the queue is full.
        """
        if not transcript or not transcript.strip():
            return
        try:
            self._speech_queue.put_nowait(TranscriptItem(transcript))
        except queue.Full:
            # Queue full means the model is behind; silently drop rather than block
            print('[ProtocolManager] Queue full, dropping transcript')

    @property
    def protocol_prediction(self):
        """PyQt signal — connect to UpdateProtocolBoxes."""
        return self._thread.protocol_prediction


# ── Background thread ─────────────────────────────────────────────────────────

class _ProtocolThread(QThread):
    # Emits a list of ProtocolPrediction objects (top-3), matching what
    # UpdateProtocolBoxes expects.
    protocol_prediction = pyqtSignal(list)

    def __init__(self, speech_queue: queue.Queue):
        super().__init__()
        self._speech_queue = speech_queue
        self._running = True
        print('[_ProtocolThread] Initialized')

    def stop(self):
        print('[_ProtocolThread] Stopping...')
        self._running = False
        self.quit()
        self.wait(5000)
        print('[_ProtocolThread] Stopped')

    def run(self):
        print('[_ProtocolThread] Started — loading protocol model...')

        # ── Load model (heavy, done here so GUI doesn't block) ────────────────
        try:
            import torch
            import numpy as np
            import yaml, re, os, time as _time
            from EMS_Agent.Interface.utils import AttrDict, onehot2p, convert_label, preprocess
            from EMS_Agent.Interface.default_sets import seed_everything, ungroup_p_node
            from EMS_Agent.Interface.EMSTinyBERTSystem import EMSTinyBERT
            from Utils import pipeline_config
        except ImportError as e:
            print(f'[_ProtocolThread] Import error: {e}')
            print('[_ProtocolThread] Protocol prediction will not be available.')
            return

        seed_everything(3407)

        config = AttrDict({
            'max_len': 512,
            'fusion': None,
            'cls': 'fc',
            'backbone': 'nlpie/tiny-clinicalbert',
            'cluster': 'group',
            'attn': 'la',
            'graph': 'hetero',
        })
        model_date = pipeline_config.protocol_model_type  # e.g. 'DKEC-TinyClinicalBERT'

        try:
            t0 = _time.time()
            model = EMSTinyBERT(config, model_date)
            print(f'[_ProtocolThread] Model loaded in {_time.time()-t0:.1f}s')

            # Warmup pass so first real inference isn't slow
            model("patient unresponsive, possible cardiac arrest")
            print('[_ProtocolThread] Warmup done — ready for transcripts')
        except Exception as e:
            print(f'[_ProtocolThread] Failed to load model: {e}')
            import traceback; traceback.print_exc()
            return

        # Accumulate transcript lines into a growing narrative, exactly like
        # the original EMSAgentInference does.
        narrative = ""

        # ── Inference loop ─────────────────────────────────────────────────────
        while self._running:
            try:
                received = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if received == 'Kill':
                print('[_ProtocolThread] Kill signal received')
                break

            if not self._running:
                break

            transcript = received.transcript.strip()
            if not transcript:
                continue

            narrative += " " + transcript
            narrative = narrative.strip()

            print(f'[_ProtocolThread] Running inference on narrative ({len(narrative)} chars)...')
            try:
                t0 = _time.time()
                protocol_arr, prob_arr, one_hot, logits = model(narrative)
                print(f'[_ProtocolThread] Inference done in {(_time.time()-t0)*1000:.0f}ms')

                # Build top-3 predictions for the GUI
                top3 = [
                    ProtocolPrediction(str(prot), float(conf))
                    for prot, conf in zip(protocol_arr, prob_arr)
                ]
                self.protocol_prediction.emit(top3)

            except Exception as e:
                print(f'[_ProtocolThread] Inference error: {e}')
                import traceback; traceback.print_exc()

        print('[_ProtocolThread] Exiting')
