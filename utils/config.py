sample_rate = 22050
window_size = 1024
overlap = 336   
seq_len = 127 # Depends on sample_rate (number of samples in audio of 4 seconds), window_size, and overlap. How did I calculate? Looked for the shape mismatch error in the logs.
mel_bins = 64

# Mappings: (as defined in metadata.csv)
# 0 = air_conditioner
# 1 = car_horn
# 2 = children_playing
# 3 = dog_bark
# 4 = drilling
# 5 = engine_idling
# 6 = gun_shot
# 7 = jackhammer
# 8 = siren
# 9 = street_music

labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}
