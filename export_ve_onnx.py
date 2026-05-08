import torch
from viterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from viterbox.models.voice_encoder.config import VoiceEncConfig

def export_voice_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Bắt đầu xuất Voice Encoder trên {device}...")

    # Khởi tạo mô hình và cấu hình
    hp = VoiceEncConfig()
    ve = VoiceEncoder(hp).to(device)
    
    # Load weights (Đảm bảo đường dẫn file ve.pt chính xác)
    ve.load_state_dict(torch.load("pretrained/ve.pt", map_location=device))
    ve.eval()

    # Dummy input: (Batch_size, Frames, Mel_channels)
    # 160 là hp.ve_partial_frames mặc định
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
