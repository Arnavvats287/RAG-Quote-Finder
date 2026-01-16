import pandas as pd
from datasets import load_dataset
import os

def prepare_data():
    dataset = load_dataset("Abirate/english_quotes")   #dataset used
    df = pd.DataFrame(dataset['train'])
    df = df[['quote', 'author', 'tags']].dropna()   #data-cleaning , removing unused columns and dropping values
    df['quote'] = df['quote'].str.strip()
    df['author'] = df['author'].str.strip()
    df['context'] = df.apply(
        lambda x: f"{x['quote']} — {x['author']} ({', '.join(x['tags'])})",            #information is combined for later uses
        axis=1
    )
    os.makedirs("saved_data", exist_ok=True)
    df.to_csv("saved_data/quotes.csv", index=False)
    print(f"Saved {len(df)} quotes to saved_data/quotes.csv")         # saved

if __name__ == "__main__":
    prepare_data()
