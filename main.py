from model_module import test_models_local as tml

# tml.calulate_threshold_wav2vec()
# output_csv = '../meta.csv'
# tml.create_results_file(output_csv)
# tml.calculate_eer_binary()

# tml.create_test_file('deep_voice_data_fake.csv','../datasets/deep_voice/FAKE')
# tml.create_test_file('deep_voice_data_real.csv','../datasets/deep_voice/REAL')

# tml.create_results_file('deep_voice_data_fake.csv', '../datasets/deep_voice/FAKE')
tml.create_results_file('deep_voice_data_fake.csv', '../datasets/deep_voice/FAKE',meso_output_csv = '../meso_deep_voice_fake.csv', wav2vec_output_csv = '../wav2vec_deep_voice_fake.csv', meso_output_finetuned = '../meso_ft_deep_voice_fake.csv'  )





