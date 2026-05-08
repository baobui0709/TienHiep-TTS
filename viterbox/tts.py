import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, List

from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from .models.t3 import T3, T3Config
from .models.t3.modules.cond_enc import T3Cond
from .models.s3gen import S3Gen, S3GEN_SR
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.voice_encoder import VoiceEncoder
from .models.tokenizers import MTLTokenizer

try:
    from soe_vinorm import SoeNormalizer
    _normalizer = SoeNormalizer()
    HAS_VINORM = True
except ImportError:
    HAS_VINORM = False
    _normalizer = None

REPO_ID = "AnhTuan89/viterbox"
WAVS_DIR = Path("wavs")
_VAD_MODEL = None
_VAD_UTILS = None

def get_vad_model():
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        try:
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True, verbose=False)
            _VAD_MODEL, _VAD_UTILS = model, utils
        except Exception as e:
            print(f"⚠️ Could not load Silero VAD: {e}")
            return None, None
    return _VAD_MODEL, _VAD_UTILS

def get_random_voice() -> Optional[Path]:
    if WAVS_DIR.exists():
        voices = list(WAVS_DIR.glob("*.wav"))
        if voices:
            import random
            return random.choice(voices)
    return None

def punc_norm(text: str) -> str:
    if len(text) == 0: return "You need to add some text for me to talk."
    if len(text) > 0 and text[0].islower(): text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    punc_to_replace = [("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "), ("—", "-"), ("–", "-"), (" ,", ","), ('"', '"'), ("'", "'")]
    for old, new in punc_to_replace: text = text.replace(old, new)
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}): text += "."
    return text

def normalize_text(text: str, language: str = "vi") -> str:
    if language == "vi" and HAS_VINORM and _normalizer is not None:
        try: return _normalizer.normalize(text)
        except Exception: return text
    return text

def _split_text_to_sentences(text: str) -> List[str]:
    parts = re.split(r'([.?!]+)', text)
    sentences, current = [], ""
    for part in parts:
        if re.match(r'([.?!]+)', part):
            current += part
            if current.strip(): sentences.append(current.strip())
            current = ""
        else:
            current = part
    if current.strip(): sentences.append(current.strip())
    return [s for s in sentences if s.strip()]

def trim_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def vad_trim(audio: np.ndarray, sr: int, margin_s: float = 0.01) -> np.ndarray:
    if len(audio) == 0: return audio
    model, utils = get_vad_model()
    if model is None: return trim_silence(audio, sr, top_db=20)
    (get_speech_timestamps, _, read_audio, *_) = utils
    wav = torch.tensor(audio, dtype=torch.float32)
    try:
        vad_sr = 16000
        if sr != vad_sr:
            wav_tensor = torch.tensor(librosa.resample(audio, orig_sr=sr, target_sr=vad_sr), dtype=torch.float32)
        else:
            wav_tensor = wav
        timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=vad_sr, threshold=0.35, min_speech_duration_ms=250, min_silence_duration_ms=100)
        if not timestamps: return trim_silence(audio, sr, top_db=25)
        cut_point = min(int(timestamps[-1]['end'] * (sr / vad_sr)) + int(margin_s * sr), len(audio))
        return audio[:cut_point]
    except Exception as e:
        return trim_silence(audio, sr, top_db=20)

def apply_fade_out(audio: np.ndarray, sr: int, fade_duration: float = 0.01) -> np.ndarray:
    if len(audio) == 0: return audio
    fade_samples = min(int(fade_duration * sr), len(audio))
    if fade_samples <= 0: return audio
    audio_copy = audio.copy()
    audio_copy[-fade_samples:] = audio_copy[-fade_samples:] * np.linspace(1.0, 0.0, fade_samples)
    return audio_copy

def apply_fade_in(audio: np.ndarray, sr: int, fade_duration: float = 0.005) -> np.ndarray:
    if len(audio) == 0: return audio
    fade_samples = min(int(fade_duration * sr), len(audio))
    if fade_samples <= 0: return audio
    audio_copy = audio.copy()
    audio_copy[:fade_samples] = audio_copy[:fade_samples] * np.linspace(0.0, 1.0, fade_samples)
    return audio_copy

def crossfade_concat(audios: List[np.ndarray], sr: int, fade_ms: int = 50, pause_ms: int = 500) -> np.ndarray:
    if not audios: return np.array([])
    if len(audios) == 1: return audios[0]
    fade_samples = int(sr * fade_ms / 1000)
    pause_samples = int(sr * pause_ms / 1000)
    result = audios[0].copy()
    for i in range(1, len(audios)):
        next_audio = audios[i]
        if pause_samples > 0: result = np.concatenate([result, np.zeros(pause_samples, dtype=result.dtype)])
        if len(result) < fade_samples or len(next_audio) < fade_samples:
            result = np.concatenate([result, next_audio])
            continue
        crossfaded = result[-fade_samples:] * np.linspace(1.0, 0.0, fade_samples) + next_audio[:fade_samples] * np.linspace(0.0, 1.0, fade_samples)
        result = np.concatenate([result[:-fade_samples], crossfaded, next_audio[fade_samples:]])
    return result

@dataclass
class TTSConds:
    t3: Union['T3Cond', dict]
    s3: dict
    ref_wav: Optional[torch.Tensor] = None
    
    @classmethod
    def load(cls, path, device):
        def to_device(x, dev):
            if isinstance(x, torch.Tensor): return x.to(dev)
            elif isinstance(x, dict): return {k: to_device(v, dev) for k, v in x.items()}
            return x
        data = torch.load(path, map_location='cpu', weights_only=False)
        t3_data, s3_data = data.get('t3', {}), data.get('gen', data.get('s3', {}))
        ref_wav = data.get('ref_wav', None)
        if isinstance(t3_data, dict) and 'speaker_emb' in t3_data:
            t3_cond = T3Cond(
                speaker_emb=to_device(t3_data['speaker_emb'], device),
                cond_prompt_speech_tokens=to_device(t3_data.get('cond_prompt_speech_tokens'), device),
                cond_prompt_speech_emb=to_device(t3_data.get('cond_prompt_speech_emb'), device) if t3_data.get('cond_prompt_speech_emb') is not None else None,
                clap_emb=to_device(t3_data.get('clap_emb'), device) if t3_data.get('clap_emb') is not None else None,
                emotion_adv=to_device(t3_data.get('emotion_adv'), device) if t3_data.get('emotion_adv') is not None else None,
            )
        else: t3_cond = to_device(t3_data, device)
        return cls(t3=t3_cond, s3=to_device(s3_data, device), ref_wav=to_device(ref_wav, device) if ref_wav is not None else None)

class Viterbox:
    def __init__(self, t3: T3, s3gen: S3Gen, ve: VoiceEncoder, tokenizer: MTLTokenizer, device: str = "cuda"):
        self.t3, self.s3gen, self.ve, self.tokenizer, self.device = t3, s3gen, ve, tokenizer, device
        self.sr = 24000
        self.conds: Optional[TTSConds] = None
        
    @classmethod
    def from_pretrained(cls, device: str = "cuda") -> 'Viterbox':
        ckpt_dir = Path(snapshot_download(repo_id=REPO_ID, repo_type="model", revision="main", allow_patterns=["ve.pt", "t3_ml24ls_v2.safetensors", "s3gen.pt", "tokenizer_vi_expanded.json", "conds.pt"], token=os.getenv("HF_TOKEN")))
        return cls.from_local(ckpt_dir, device)
    
    @classmethod
    def from_local(cls, ckpt_dir: Union[str, Path], device: str = "cuda") -> 'Viterbox':
        ckpt_dir = Path(ckpt_dir)
        
        # Load Voice Encoder (Nhận diện file ONNX ưu tiên)
        onnx_ve_path = Path("pretrained/ve.onnx")
        if onnx_ve_path.exists():
            ve = VoiceEncoder(onnx_path=str(onnx_ve_path)).to(device).eval()
        else:
            ve = VoiceEncoder()
            ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", map_location='cpu' if device == "mps" else device, weights_only=True))
            ve.to(device).eval()
        
        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_ml24ls_v2.safetensors")
        if "model" in t3_state: t3_state = t3_state["model"][0]
        
        for k in ["text_emb.weight", "text_head.weight"]:
            if k in t3_state:
                old_w = t3_state[k]
                if old_w.shape[0] != t3.hp.text_tokens_dict_size:
                    new_w = torch.zeros((t3.hp.text_tokens_dict_size, old_w.shape[1]), dtype=old_w.dtype)
                    min_rows = min(old_w.shape[0], new_w.shape[0])
                    new_w[:min_rows] = old_w[:min_rows]
                    if new_w.shape[0] > min_rows: nn.init.normal_(new_w[min_rows:], mean=0.0, std=0.02)
                    t3_state[k] = new_w
        
        t3.load_state_dict(t3_state)
        t3.to(device).eval()
        
        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", map_location='cpu' if device == "mps" else device, weights_only=True))
        s3gen.to(device).eval()
        
        tokenizer = MTLTokenizer(str(ckpt_dir / "tokenizer_vi_expanded.json"))
        model = cls(t3, s3gen, ve, tokenizer, device)
        if (ckpt_dir / "conds.pt").exists(): model.conds = TTSConds.load(ckpt_dir / "conds.pt", device)
        return model
    
    def prepare_conditionals(self, audio_prompt: Union[str, Path, torch.Tensor], exaggeration: float = 0.5):
        s3gen_ref_wav, _ = librosa.load(str(audio_prompt), sr=S3GEN_SR, mono=True) if isinstance(audio_prompt, (str, Path)) else (audio_prompt.cpu().numpy().squeeze(), None)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[:S3GEN_SR * 10]
        
        with torch.inference_mode():
            s3_cond = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
            t3_cond_prompt_tokens = None
            if plen := self.t3.hp.speech_cond_prompt_len:
                t3_cond_prompt_tokens = torch.atleast_2d(self.s3gen.tokenizer.forward([ref_16k_wav[:S3_SR * 10]], max_len=plen)[0]).to(self.device)
            
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)).mean(axis=0, keepdim=True).to(self.device)
            t3_cond = T3Cond(speaker_emb=ve_embed, cond_prompt_speech_tokens=t3_cond_prompt_tokens, emotion_adv=exaggeration * torch.ones(1, 1, 1)).to(device=self.device)
        self.conds = TTSConds(t3=t3_cond, s3=s3_cond, ref_wav=torch.from_numpy(s3gen_ref_wav).unsqueeze(0))
        return self.conds
    
    def _generate_single(self, text: str, language: str, cfg_weight: float, temperature: float, top_p: float, repetition_penalty: float) -> np.ndarray:
        text_tokens = self.tokenizer.text_to_tokens(punc_norm(text), language_id=language).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=self.t3.hp.start_text_token), (0, 1), value=self.t3.hp.stop_text_token)

        with torch.inference_mode(), torch.autocast(device_type='cuda' if self.device == 'cuda' else 'mps', dtype=torch.float16, enabled=self.device in ['cuda', 'mps']):
            speech_tokens = drop_invalid_tokens(self.t3.inference(
                t3_cond=self.conds.t3, text_tokens=text_tokens, max_new_tokens=1000, temperature=temperature,
                cfg_weight=cfg_weight, repetition_penalty=repetition_penalty, top_p=top_p
            )[0])
            if len(speech_tokens) > 1: speech_tokens = speech_tokens[:-1]
            wav, _ = self.s3gen.inference(speech_tokens=speech_tokens.to(self.device), ref_dict=self.conds.s3)
        return wav[0].cpu().numpy()
    
    def generate(self, text: str, language: str = "vi", audio_prompt: Optional[Union[str, Path, torch.Tensor]] = None, exaggeration: float = 0.5, cfg_weight: float = 0.5, temperature: float = 0.8, top_p: float = 1.0, repetition_penalty: float = 2.0, split_sentences: bool = True, crossfade_ms: int = 50, sentence_pause_ms: int = 500) -> torch.Tensor:
        if audio_prompt is not None: self.prepare_conditionals(audio_prompt, exaggeration)
        elif self.conds is None:
            if rv := get_random_voice(): self.prepare_conditionals(rv, exaggeration)
            else: raise ValueError("No reference audio!")
        
        text = normalize_text(text, language)
        if split_sentences:
            sentences = _split_text_to_sentences(text) or [text]
            audio_segments = [apply_fade_in(apply_fade_out(vad_trim(self._generate_single(s, language, cfg_weight, temperature, top_p, repetition_penalty), self.sr, 0.05), self.sr, 0.01), self.sr, 0.005) for i, s in enumerate(sentences) if print(f"  [{i+1}/{len(sentences)}] {s[:50]}...") or len(self._generate_single(s, language, cfg_weight, temperature, top_p, repetition_penalty))]
            return torch.from_numpy(apply_fade_out(crossfade_concat(audio_segments, self.sr, fade_ms=crossfade_ms, pause_ms=sentence_pause_ms), self.sr, 0.015)).unsqueeze(0) if audio_segments else torch.zeros(1, self.sr)
        return torch.from_numpy(self._generate_single(text, language, cfg_weight, temperature, top_p, repetition_penalty)).unsqueeze(0)
    
    def save_audio(self, audio: torch.Tensor, path: Union[str, Path], trim_silence: bool = True):
        import soundfile as sf
        audio_np = audio[0].cpu().numpy()
        if trim_silence: audio_np, _ = librosa.effects.trim(audio_np, top_db=30)
        sf.write(str(path), audio_np, self.sr)
