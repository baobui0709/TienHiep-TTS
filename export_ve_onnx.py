import torch
import os
import warnings
from huggingface_hub import hf_hub_download
from viterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from viterbox.models.voice_encoder.config import VoiceEncConfig

# Bỏ qua các cảnh báo không cần thiết của PyTorch
warnings.filterwarnings("ignore")

def export_voice_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Bắt đầu xuất Voice Encoder trên {device}...")

    os.makedirs("pretrained", exist_ok=True)
    ve_path = "pretrained/ve.pt"

    if not os.path.exists(ve_path):
        print("⏳ Đang tải weights ve.pt từ HuggingFace...")
        ve_path = hf_hub_download(repo_id="AnhTuan89/viterbox", filename="ve.pt", local_dir="pretrained")

    hp = VoiceEncConfig()
    hp.flatten_lstm_params = False 
    
    ve = VoiceEncoder(hp).to(device)
    ve.load_state_dict(torch.load(ve_path, map_location=device))
    ve.eval()

    # [QUAN TRỌNG] Hack để chặn lỗi "FakeTensor" của LSTM trên PyTorch 2.x
    ve.lstm.flatten_parameters = lambda: None

    dummy_mels = torch.randn(1, hp.ve_partial_frames, hp.num_mels, device=device)

    # Sử dụng JIT Trace để ép PyTorch dùng engine ổn định thay vì Dynamo
    print("Đang đóng băng đồ thị (JIT Trace)...")
    traced_model = torch.jit.trace(ve, dummy_mels)

    onnx_path = "pretrained/ve.onnx"
    torch.onnx.export(
        traced_model,
        dummy_mels,
        onnx_path,
        export_params=True,
        opset_version=18, # Nâng lên opset 18 theo chuẩn mới
        do_constant_folding=True,
        input_names=['mels'],
        output_names=['speaker_embed'],
        dynamic_axes={
            'mels': {0: 'batch_size', 1: 'frames'}, 
            'speaker_embed': {0: 'batch_size'}
        }
    )
    print(f"✅ Đã xuất thành công: {onnx_path}")

if __name__ == "__main__":
    export_voice_encoder()
