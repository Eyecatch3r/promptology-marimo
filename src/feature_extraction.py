import polars as pl
import elfen

import argparse

def extract_features(df: pl.DataFrame,
                     text_column: str,
                     language: str,
                     model: str,
                     save_path: str
                     ) -> None:
    """
    Extracts text features using the Elfen library.

    Parameters:
    df (pl.DataFrame): Input DataFrame containing text data.
    text_column (str): Name of the column containing text data.
    language (str): Language of the text data.
    model (str): Pre-trained model to use for feature extraction.
    save_path (str): Path to save the extracted features CSV file.

    Returns:
     None
    """
    # Initialize elfen extractor
    extractor = elfen.Extractor(data = df,
                                text_column = text_column,
                                language = language,
                                model = model)
    
    # Extract features
    extractor.extract_features()

    # save the dataframe with features as csv
    extractor.write_csv(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text features "
                                                 "using Elfen.")
    parser.add_argument("input_csv", type=str,
                        help="Path to the input CSV file.")
    parser.add_argument("text_column", type=str,
                        help="Name of the text column in the CSV file.")
    parser.add_argument("language", type=str,
                        help="Language of the text data (e.g., 'en' for "
                             "English).")
    parser.add_argument("model", type=str,
                        help="Pre-trained model to use for feature "
                             "extraction (e.g., 'bert-base-uncased').")
    parser.add_argument("output_csv", type=str,
                        help="Path to save the output CSV file with "
                             "extracted features.")

    args = parser.parse_args()

    # Load data
    df = pl.read_csv(args.input_csv)

    # fill NaN values in text column with empty string
    df = df.with_columns(
        pl.col(args.text_column).fill_null("")
    )

    # Extract features
    extract_features(df,
                     args.text_column,
                     args.language,
                     args.model,
                     args.output_csv)

