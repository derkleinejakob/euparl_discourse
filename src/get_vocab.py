import pandas as pd
import re


def get_vocab(df: pd.DataFrame, text_column: str = "translatedText") -> list[str]:
    words = df[text_column].str.split().explode()  # take speeches and split on whitespace
    words = words.apply(lambda word: re.sub(r'\W+', '', word).lower())  # normalize words
    vocab = list(set(words))  # get unique
    return vocab


if __name__ == '__main__':
    IN_PATH = "data/parllaw/migration.parquet"
    OUT_PATH = "data/parllaw/vocab.txt"

    df = pd.read_parquet(IN_PATH)

    vocab = get_vocab(df)

    print(f"Vocab length: {len(vocab)}")

    with open(OUT_PATH, 'w') as f:
        f.write("\n".join(vocab))
 