import pandas as pd

def main():
    clinical_text_df = pd.read_excel("data/eimara_annotations.xls")

    # pycontextnlp method
    negations_pycontextnlp(clinical_text_df)


if __name__ == "__main__":
    main()