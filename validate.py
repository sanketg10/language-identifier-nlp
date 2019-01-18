import json
import os

all_languages = set(['th', 'de', 'ur', 'no', 'sv', 'ja', 'eo', 'ms', 'sr', 'vi', 'fi', 'sl', 'he', 'ce', 'ro', 'ka', 'el', 'es', 'lorem', 'uz', 'la', 'bg', 'ca', 'az', 'fa', 'pl', 'hr', 'et', 'ar', 'nl', 'uk', 'war', 'id', 'da', 'hy', 'it', 'gl', 'hu', 'sk', 'pt', 'ko', 'zh', 'ceb', 'hi', 'cs', 'vo', 'ru', 'kk', 'lt', 'nn', 'en', 'be', 'fr', 'tr', 'eu', 'sh'])

def run_basic_validations(validation_file):

    assert os.path.exists(validation_file), 'Prediction file predictions.json.txt is missing'

    with open(validation_file, 'rt') as pred_file:
        lines = pred_file.readlines()
        assert len(lines) == 124609, 'Did not submit correct number of predictions'

        try:
            json.loads(lines[0])
        except Exception as e:
            print('Failed to parse json')
            raise e

        sample = json.loads(lines[123])

        assert 'classification' in sample, 'correct label missing'
        assert sample['classification'] in all_languages

    print('Formatting looks good, hope your model performs well!')


if __name__ == '__main__':
    run_basic_validations('predictions.json.txt')
