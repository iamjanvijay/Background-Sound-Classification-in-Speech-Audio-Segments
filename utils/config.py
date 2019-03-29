sample_rate = 22050
window_size = 1024
overlap = 336   
seq_len = 127 # Depends on sample_rate (number of samples in audio of 4 seconds), window_size, and overlap. How did I calculate? Looked for the shape mismatch error in the logs.
mel_bins = 64

seed = 1234

dataset = 'UrbanSound8K'

if dataset == 'UrbanSound8K':
	# Mappings: (as defined in metadata.csv) UrbanSound8K Dataset
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
	metadata_delimiter = ','

elif dataset == 'YBSS-150':
	# Mappings: (as defined in metadata.csv) YBSS-150 Dataset
	# 0 = auto
	# 1 = crowd
	# 2 = formula_1
	# 3 = guitar
	# 4 = helicopter
	# 5 = tap_water
	labels = ['auto', 'crowd', 'formula_1', 'guitar', 'helicopter', 'tap_water']
	metadata_delimiter = '|'

lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}