import torch
import random


def RandomMask(x, ratio):
    # x: [batch,channels,seq_len]
    # randomly mask some data as 0
    mask = (torch.rand(x.shape) > ratio)
    x *= mask
    return x


def CutOut(x, ratio):
    # x: [batch,channels,seq_len]
    # cuts out continuous segments of the sequence
    cut_length = int(x.shape[-1] * ratio)
    assert cut_length > 0
    for b in range(x.shape[0]):
        start = torch.randint(0, x.shape[-1] - cut_length + 1, (1,))
        x[b, :, start:start + cut_length] = 0
    return x


def JitterScale(x, jitter_range=0.1, scale_range=(0.5, 2)):
    # x: [batch,channels,seq_len]
    # Add random noise and apply random scaling
    jitter = torch.randn_like(x.float()) * jitter_range
    scale = torch.rand(x.shape[0], 1, 1) * (scale_range[1] - scale_range[0]) + scale_range[0]
    return (x + jitter) * scale


def PermutationJitter(x, min_len, max_segment, jitter_range=0.1):
    # permutation includes splitting the signal into a random number of segments with a maximum of M and randomly shuffling them
    batch, channels, seq_len = x.shape
    x_out = x.clone()
    max_possible_segments = seq_len // min_len
    jitter = torch.randn_like(x.float()) * jitter_range

    for b in range(batch):
        # Calculate number of segments based on min_len
        num_segments = random.randint(2, min(max_segment, max_possible_segments))

        # Generate valid segment points respecting min_len
        valid_points = []
        current = min_len
        while current < seq_len - min_len and len(valid_points) < num_segments - 1:
            point = random.randint(current, min(current + min_len, seq_len - min_len))
            valid_points.append(point)
            current = point + min_len

        valid_points.sort()
        segments = []

        start = 0
        for end in valid_points:
            segments.append(x_out[b, :, start:end].clone())
            start = end
        segments.append(x_out[b, :, start:].clone())

        random.shuffle(segments)

        current_pos = 0
        for segment in segments:
            segment_length = segment.shape[-1]
            x_out[b, :, current_pos:current_pos + segment_length] = segment
            current_pos += segment_length

    return x_out + jitter


if __name__ == '__main__':
    a = torch.randint(1, 5, (2, 3, 20), requires_grad=False)
    print(a)
    # mask = RandomMask(a.clone(), 0.3)
    # print(mask)
    # cut_out = CutOut(a.clone(), 0.3)
    # print(cut_out)
    # print(JitterScale(a.clone()))
    print(PermutationJitter(a.clone(), 3, 5, 0))
