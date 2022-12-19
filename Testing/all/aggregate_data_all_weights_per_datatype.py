
data_types = ['full_deformed', 'partial_deformed', 'full_non_deformed', 'partial_non_deformed']
feature_extractors = ['fcgf', 'kpfcn']
training_data = {
    'full_deformed' : 2, 
    'partial_deformed' : 1
}

for data_type in data_types:
    