
message_length = 79 # number symbol
baud = 6.26 # baud 1 / seconds
spacing = 6.25 #hz
mary = 8 # 8 fsk levels

symbol_time = message_length / baud
frequencies = spacing * mary

# if we make an epoch from direct
sdr_rate = 140000000 # real
epoch = 2/baud
epoch_samples = sdr_rate * epoch

# what about based on frequency resolution
freqspbin = spacing / 2
freqs_samples = (sdr_rate / 2) / freqspbin

power = 2
while 2**power < freqs_samples:
    power += 1

freqpbin_calc = (sdr_rate/2) / 2**power
epoch_calc = 2**power / sdr_rate / 2
bins_per_baud = spacing / freqpbin_calc

print(f"Symbol time is {symbol_time} seconds")
print(f"Frequencies are {frequencies}")
print(f"Epoch is {epoch} with {epoch_samples} samples")
print(f"Freqs per bin {freqspbin} sdr rate {freqs_samples}")
print(f"Closest power 2 {power} size {2**power}")
print(f"power {power} gives freq resolution {freqpbin_calc} epoch {epoch_calc} vs {1/baud}")
print(f"This is {bins_per_baud} bins per baud")
# 33554432 gives us 3 bins per baud which is perfect





