import pandas as pd


def transcript_iterator(clinical_text_df):
    for index, row in clinical_text_df.iterrows():
        # if the row has a NaN value in the transcripts column, skip it
        if pd.isna(clinical_text_df.iloc[index, 0]):
            continue

        print("Transcript " + str(index) + ", row " + str(index + 2) + ":")
        transcript_to_process = row[0]
        sentences = transcript_to_sentences_of_tokens(transcript_to_process, False)

        for sentence in sentences:
            print(sentence)

        print('\n')


def transcript_to_sentences_of_tokens(transcript_to_process, return_tokens=True):
    # first split the transcript by whitespace into a list
    list_tokens = transcript_to_process.split()

    # if a tokens last character is ':' or '-', add a special character ^ to the end of the token before it
    # indicating end of sentence.
    for i in range(len(list_tokens)):
        if (list_tokens[i][-1] == ':' or list_tokens[i][-1] == '-') \
                and (i != 0) \
                and (list_tokens[i - 1][-1] != '.') \
                and (list_tokens[i - 1][-1] != '!') \
                and (list_tokens[i - 1][-1] != '?'):
            list_tokens[i - 1] = list_tokens[i - 1] + '^'

    # a list of concatenations that do not signify end of sentence.
    list_concatenations = ['pt.', 'st.', 'dr.']

    # divide the tokens into their own individual lists by looking for punctuation characters
    # default punctuation characters are '.', '!', '?',
    list_sentences = []

    sentence = []
    while len(list_tokens) != 0:
        if (list_tokens[0][-1] == '.'
            or list_tokens[0][-1] == '!'
            or list_tokens[0][-1] == '?'
            or list_tokens[0][-1] == '^') \
                and list_tokens[0].lower() not in list_concatenations:

            # special character check, remove if it's there
            if list_tokens[0][-1] == '^':
                list_tokens[0] = list_tokens[0][:-1]

            sentence.append(list_tokens.pop(0))
            list_sentences.append(sentence)
            sentence = []
        else:
            sentence.append(list_tokens.pop(0))

    # if return_tokens == True, then return the sentences as lists of tokens
    # otherwise, return as individual strings
    if return_tokens:
        return list_sentences
    else:
        list_sentences_joined = []
        for sentence in list_sentences:
            list_sentences_joined.append(' '.join(sentence))
        return list_sentences_joined


def main():
    clinical_text_df = pd.read_excel("data/eimara_annotations.xls")
    transcript_iterator(clinical_text_df)


if __name__ == "__main__":
    main()
