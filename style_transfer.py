import pandas as pd
import yaml
import re


def load_replacements():
    with open(r'data/replacements.yml') as file:
        return list(yaml.load_all(file, Loader=yaml.FullLoader))


def style_transfer(clinical_text_df):
    converted_transcripts = []
    replacements = load_replacements()
    for index, row in clinical_text_df.iterrows():
        transcript_to_process = row[0]
        for replacement in replacements:
            find_regex = replacement['Find']
            replacement_string = replacement['Replacement']
            transcript_to_process = re.sub(find_regex, replacement_string, transcript_to_process)
        converted_transcripts.append(transcript_to_process)
    clinical_text_df['Converted'] = converted_transcripts
    return clinical_text_df


def main():
    clinical_text_df = pd.read_csv('data/eimara_transcripts.csv')
    clinical_text_df = style_transfer(clinical_text_df)
    clinical_text_df.to_csv(path_or_buf='data/eimara_transcripts.csv', index=False)


if __name__ == "__main__":
    main()
