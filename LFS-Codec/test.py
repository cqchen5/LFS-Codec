
import torch

from moshi.models import loaders, LMGen

mimi_weight = 'ckpt/tokenizer-e351c8d8-checkpoint125.safetensors'

mimi = loaders.get_mimi(mimi_weight, device='cpu')
mimi.set_num_codebooks(8)  # up to 32 for mimi, but limited to 8 for moshi.

wav = torch.randn(1, 1, 24000 * 10)  # should be [B, C=1, T]
with torch.no_grad():
    codes = mimi.encode(wav)  # [B, K = 8, T]
    print(codes.shape)
    decoded = mimi.decode(codes)
    print(decoded.shape)
    # Supports streaming too.
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    all_codes = []
    with mimi.streaming(batch_size=1):
        for offset in range(0, wav.shape[-1], frame_size):
            frame = wav[:, :, offset: offset + frame_size]
            codes = mimi.encode(frame)
            assert codes.shape[-1] == 1, codes.shape
            all_codes.append(codes)

## WARNING: When streaming, make sure to always feed a total amount of audio that is a multiple
#           of the frame size (1920), otherwise the last frame will not be complete, and thus
#           will not be encoded. For simplicity, we recommend feeding in audio always in multiple
#           of the frame size, so that you always know how many time steps you get back in `codes`.

# Now if you have a GPU around.
mimi.cuda()
moshi_weight = 'ckpt/model.safetensors'
moshi = loaders.get_moshi_lm(moshi_weight, device='cuda')
lm_gen = LMGen(moshi, temp=0.8, temp_text=0.7)  # this handles sampling params etc.
out_wav_chunks = []
# Now we will stream over both Moshi I/O, and decode on the fly with Mimi.
with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
    for idx, code in enumerate(all_codes):
        tokens_out = lm_gen.step(code.cuda())
        # tokens_out is [B, 1 + 8, 1], with tokens_out[:, 1] representing the text token.
        if tokens_out is not None:
            wav_chunk = mimi.decode(tokens_out[:, 1:])
            out_wav_chunks.append(wav_chunk)
        print(idx, end='\r')
out_wav = torch.cat(out_wav_chunks, dim=-1)