import torch
import os
from huggingface_hub import hf_hub_download
from viterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from viterbox.models.voice_encoder.config import VoiceEncConfig

def export_voice_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Bắt đầu xuất Voice Encoder trên {device}...")

    # Đảm bảo thư mục pretrained tồn tại
    os.makedirs("pretrained", exist_ok=True)
    ve_path = "pretrained/ve.pt"

    # Tự động tải ve.pt từ HuggingFace nếu chưa có
    if not os.path.exists(ve_path):
        print("⏳ Đang tải weights ve.pt từ HuggingFace (AnhTuan89/viterbox)...")
        ve_path = hf_hub_download(
            repo_id="AnhTuan89/viterbox",
            filename="ve.pt",
            local_dir="pretrained"
        )
        print("✅ Đã tải weights thành công!")

    # Khởi tạo mô hình và cấu hình
    hp = VoiceEncConfig()
    ve = VoiceEncoder(hp).to(device)
    
    # Load weights
    ve.load_state_dict(torch.load(ve_path, map_location=device))
    ve.eval()

    # Dummy input: (Batch_size, Frames, Mel_channels)
    dummy_mels = torch.randn(1, hp.ve_partial_frames, hp.num_mels, device=device)

    # Xuất ONNX
    onnx_path = "pretrained/ve.onnx"
    torch.onnx.export(
        ve,
        dummy_mels,
        onnx_path,
        export_params=True,
        opset_version=17,
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
