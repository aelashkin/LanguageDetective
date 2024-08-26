import pandas as pd
import re
import nltk
#import spacy
import os


def load_data(parquet_file, xlsx_files):
    if os.path.exists(parquet_file):
        df = load_dataframe_from_parquet(parquet_file)
    else:
        dfs = [pd.read_excel(xlsx_file) for xlsx_file in xlsx_files]
        df = pd.concat(dfs, ignore_index=True)
        df['text_corrected'] = df['text_corrected'].astype(str)
        save_dataframe_to_parquet(df, "data.parquet")
    return df


def save_dataframe_to_parquet(df, file_path):
    df.to_parquet(file_path, index=False)


def load_dataframe_from_parquet(file_path):
    return pd.read_parquet(file_path)


def save_dataframe_to_csv(df, file_path):
    df.to_csv(file_path, index=False)


def load_dataframe_from_csv(file_path):
    return pd.read_csv(file_path)


def load_data_excel_dataframe(*file_paths):
    dfs = [pd.read_excel(file) for file in file_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


#######################################################################


def find_duplicate_text_corrected_ids(df, text_fixed_column, id_column):
    # Find duplicate values in the text_fixed column
    duplicate_text_fixed = df[df.duplicated(subset=[text_fixed_column], keep=False)]

    # Sort the duplicates by text_fixed to group identical values together
    duplicate_text_fixed_sorted = duplicate_text_fixed.sort_values(by=[text_fixed_column])

    # Select only the relevant columns (ID and text_fixed)
    result_df = duplicate_text_fixed_sorted[[id_column, text_fixed_column]]

    if not result_df.empty:
        print("Duplicate text_fixed values and their corresponding IDs:")
        print(result_df.to_string(index=False))
    else:
        print("No duplicate text_fixed values found.")


def remove_duplicate_text_corrected_rows(df, text_fixed_column):
    df_unique = df.drop_duplicates(subset=[text_fixed_column], keep='first')
    return df_unique


#######################################################################


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\.{3,}', '.', text)  # Replace multiple dots with a dot
    text = re.sub(r'\!{2,}', '!', text)  # Replace multiple exclamation marks with a single exclamation mark
    text = re.sub(r'\?{2,}', '?', text)  # Replace multiple question marks with a single question mark

    text = remove_language_hints(text)
    text = text.lower()
    text = text.strip()
    return text


def remove_language_hints(text):
    countries = ['Brazil', 'Taiwan', 'Mexico', 'China', 'Russia', 'Turkey', 'Italy', 'France', 'German',
                 'Saudi Arabia', 'Japan']
    nationalities = ['brazilian', 'mexican', 'chinese', 'taiwanese', 'russian', 'turkish', 'italian',
                     'french', 'german', 'saudi arabian', 'japanese']
    capitals = ['Brasilia', 'Mexico City', 'Mexico', 'Beijing', 'Taipei City', 'Taipei', 'Moscow', 'Berlin',
                'Riyadh', 'Rome', 'Tokyo', 'Ankara']

    country_pattern = re.compile(r'\b(' + '|'.join(re.escape(country) for country in countries) + r')\b', re.IGNORECASE)
    nationality_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(nationality) for nationality in nationalities) + r')\b', re.IGNORECASE)
    capital_pattern = re.compile(r'\b(' + '|'.join(re.escape(capital) for capital in capitals) + r')\b', re.IGNORECASE)

    text = country_pattern.sub('country', text)
    text = nationality_pattern.sub('nationality', text)
    text = capital_pattern.sub('capital', text)
    return text


#######################################################################


#   doesn't work well for both columns - creates mismatch in number of sentences
def split_into_sentences_nltk(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    return sentences


#def split_into_sentences_spacy(text):
#    nlp = spacy.load('en_core_web_sm')
#    doc = nlp(text)
#    sentences = [sent.text for sent in doc.sents]
#    return sentences


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    # Remove leading and trailing whitespace including newlines and tabs from each sentence
    sentences = [sentence.strip() for sentence in sentences]
    return sentences


#   to check for text and text_corrected sentence separation mismatches
def find_mismatched_sentence_counts(df, text_column, text_corrected_column):
    # Clean the text columns
    df['cleaned_text'] = df[text_column].apply(clean_text)
    df['cleaned_text_corrected'] = df[text_corrected_column].apply(clean_text)

    # Split the cleaned text into sentences
    df['sentences'] = df['cleaned_text'].apply(split_into_sentences)
    df['sentences_corrected'] = df['cleaned_text_corrected'].apply(split_into_sentences)

    # Identify mismatched sentence counts
    df['len_sentences'] = df['sentences'].apply(len)
    df['len_sentences_corrected'] = df['sentences_corrected'].apply(len)
    mismatched_df = df[df['len_sentences'] != df['len_sentences_corrected']]

    return mismatched_df


#######################################################################

def clean_and_split_texts(df, text_column, text_corrected_column, split_function):
    # remove a broken row
    df = df[df['writing_id'] != 758131].copy()

    df['text_id'] = range(1, len(df) + 1)

    # clean text, for example remove "......"
    df['cleaned_text'] = df[text_column].apply(clean_text)
    df['cleaned_text_corrected'] = df[text_corrected_column].apply(clean_text)

    # Split the cleaned text into sentences
    df['sentences'] = df['cleaned_text'].apply(split_function)
    df['sentences_corrected'] = df['cleaned_text_corrected'].apply(split_function)

    # Explode the sentences into separate rows
    df = df.explode(['sentences', 'sentences_corrected']).reset_index(drop=True)

    # Ensure that the number of sentences match
    df = df[df['sentences'].str.len() > 0]
    df = df[df['sentences_corrected'].str.len() > 0]

    return df

def clean_and_split_corrected_only(df, text_corrected_column, split_function):
    #   remove a broken row
    df = df[df['writing_id'] != 758131].copy()

    # Add a unique identifier for each text
    df['text_id'] = range(1, len(df) + 1)

    #   clean text, for example remove "......"
    df['cleaned_text_corrected'] = df[text_corrected_column].apply(clean_text)

    #    Split the cleaned text into sentences
    df['sentences_corrected'] = df['cleaned_text_corrected'].apply(split_function)

    #    Explode the sentences into separate rows
    df = df.explode(['sentences_corrected']).reset_index(drop=True)
    return df




#######################################################################


def get_clean_sentences_and_labels_only_corrected():
    if os.path.exists('clean_corrected.parquet'):
        res = load_dataframe_from_parquet('clean_corrected.parquet')
        return res
    else:
        combined_df = load_data("data.parquet", ["Final database (main prompts).xlsx",
                                                 "Final database (alternative prompts).xlsx"])
        unique_df = remove_duplicate_text_corrected_rows(combined_df, 'text_corrected')
        nltk.download('punkt_tab')
        cleaned_split_df = clean_and_split_corrected_only(unique_df,
                                                          'text_corrected', split_into_sentences_nltk)
        res = cleaned_split_df[['text_id', 'sentences_corrected', 'l1']].copy()
        save_dataframe_to_parquet(res, "clean_corrected.parquet")
        return res


def get_clean_sentences_and_labels_both():
    if os.path.exists('clean_both.parquet'):
        res = load_dataframe_from_parquet('clean_both.parquet')
        return res
    else:
        combined_df = load_data("data.parquet", ["Final database (main prompts).xlsx",
                                                 "Final database (alternative prompts).xlsx"])
        unique_df = remove_duplicate_text_corrected_rows(combined_df, 'text_corrected')
        nltk.download('punkt_tab')
        cleaned_split_df = clean_and_split_texts(unique_df, 'text',
                                                 'text_corrected', split_into_sentences)
        res = cleaned_split_df[['text_id', 'sentences', 'sentences_corrected', 'l1']].copy()
        save_dataframe_to_parquet(res, "clean_both.parquet")
        return res

def get_clean_sentences_and_labels_wo_split():
    if os.path.exists('clean_corrected_nosplit.parquet'):
        res = load_dataframe_from_parquet('clean_corrected_nosplit.parquet')
        return res
    else: 
        combined_df = load_data("data.parquet", ["Final database (main prompts).xlsx",
                                                 "Final database (alternative prompts).xlsx"])
        
        cleaned_df = remove_duplicate_text_corrected_rows(combined_df, 'text_corrected')
        cleaned_df = cleaned_df[cleaned_df['writing_id'] != 758131].copy()
        cleaned_df['cleaned_text_corrected'] = cleaned_df['text_corrected'].apply(clean_text).copy()
        res = cleaned_df[['cleaned_text_corrected', 'l1']].copy()
        save_dataframe_to_parquet(res, "clean_corrected_nosplit.parquet")
        return res


def get_clean_sentences_and_labels_wo_split_new(*file_paths, text_type='text_corrected'):
    if os.path.exists('clean_corrected_nosplit.parquet'):
        res = load_dataframe_from_parquet('clean_corrected_nosplit.parquet')
        return res
    else:
        combined_df = load_data_excel_dataframe(*file_paths)
        cleaned_df = remove_duplicate_text_corrected_rows(combined_df, 'text_corrected')
        cleaned_df['cleaned_text_corrected'] = cleaned_df[text_type].apply(clean_text).copy()
        res = cleaned_df[['cleaned_text_corrected', 'l1']].copy()
        save_dataframe_to_parquet(res, "clean_corrected_nosplit.parquet")
        return res


def get_clean_sentences_and_labels_text(*file_paths, text_type='text'):
    if os.path.exists('clean_text_nosplit.parquet'):
        res = load_dataframe_from_parquet('clean_text_nosplit.parquet')
        return res
    else:
        combined_df = load_data_excel_dataframe(*file_paths)
        cleaned_df = remove_duplicate_text_corrected_rows(combined_df, text_type)
        cleaned_df['cleaned_text'] = cleaned_df[text_type].apply(clean_text).copy()
        res = cleaned_df[['cleaned_text', 'l1']].copy()
        save_dataframe_to_parquet(res, "clean_text_nosplit.parquet")
        return res


