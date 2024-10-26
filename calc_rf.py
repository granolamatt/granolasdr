am_freqs = [
            [1.8,2.0],
            [3.5,4.0],
            [5.3305, 5.4355],
            [7.0,7.3],
            [10.1,10.15],
            [14.0,14.35],
            [18.068, 18.168],
            [21.0, 21.45],
            [24.89, 24.99],
            [28,29.7]
]


message_length = 79 # number symbol
baud = 6.26 # baud 1 / seconds
spacing = 6.25 #hz
mary = 8 # 8 fsk levels

symbol_time = message_length / baud
frequencies = spacing * mary

# if we make an epoch from direct
sdr_rate = 140000000 # real
power = 20
large_fft = 2**20 # makes 2**19 + 1 complex
complex_fft = 2**19 # this is the complex numbers we get out
epoch = large_fft / sdr_rate / 2 # in time

complex_rate = (complex_fft/epoch) / 2
freqsperbin = complex_rate / complex_fft
print(f"sdr epoch {epoch} complex rate {complex_rate} freqsperbin {freqsperbin}")

total_bins = 0
mybins = 0
# now calculate the bins needed
bins = []
for freq in am_freqs:
    start = int(freq[0] * 1e6 / freqsperbin) - 1
    stop = int(freq[1] * 1e6 / freqsperbin) + 1
    if (start % 4) != 0:
        start -= (start % 4)
        if start < 0:
            start = 0
    if (stop % 4) != 0:
        stop = (stop + 4) - (stop % 4)
        if stop > complex_fft:
            stop = complex_fft
    mybins = stop - start
    total_bins += mybins
    print(f"{freq[0]} start {start} - {freq[1]} stop {stop} bins {total_bins} bw {mybins * freqsperbin}")
    bins.append([start, stop, stop-start])

hf_fft = int(total_bins)
p = 1024
while p < hf_fft:
    p *= 2
hf_fft = p
# if (hf_fft % 4) != 0:
#     hf_fft = (hf_fft + 4) - (hf_fft % 4)

hf_rate = (hf_fft/epoch) / 2
print(f"HF fft {hf_fft} rate is {hf_rate}")

oversample = 4 # maybe 8 but 4 fits great into fft land
hfepoch = 1/baud
rhf_fft = int(hfepoch * hf_rate * oversample)
if (rhf_fft % 4) != 0:
    rhf_fft = (rhf_fft + 4) - (rhf_fft % 4)
rhfepoch = rhf_fft / hf_rate
hf_res = hf_rate / rhf_fft

print(f"HF fft is {rhf_fft} epoch {rhfepoch} vs {1/baud} total {message_length/baud} resolution {hf_res}")

print(f"Summary put \ngm::rx888::rx888::NLARGE = {large_fft}")
print(f"gm::cuda::HFChannelizer::fft_length = {hf_fft}")
print(f"gm::cuda::HFChannelizer::rfft_length = {rhf_fft}")
print("gm::cuda::HFChannelizer::frequencies")
print("const std::vector<std::vector<uint32_t>> bins = {")
for n in bins:
    print("{"+f"{n[0]},{n[1]},{n[2]}" + "},")
print("};")
