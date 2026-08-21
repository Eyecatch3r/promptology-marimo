from itertools import chain
import pandas as pd
from prompt_templates import generate_prompt


sociodemographic_infos = ['gender', 'age', 'residence', 'civil_status', 'denomination', 'education', 'political_spectrum']
corpus = pd.read_csv("./corpus.csv")  # 12758 entries


def compute_combinations(row):
    '''
    For a given row, generate all possible combinations of sociodemographics and issue+stance pairs. Skip NaN sociodemographics entries.
    We use all sociodemographics given in the corpus ("important political issues" is not used for now).
    '''
    combis = []
    for socdem in sociodemographic_infos:
        # We skip those sociodemographic entries that are empty or not defined ...
        if not (pd.isnull(row[socdem]) or row[socdem] == "Keine Geschlecht"):
            combis.append([row['question'], row['stance'], socdem, row[socdem]])
    # add a "no sociodemographics" case for all entries
    combis.append([row['question'], row['stance'], None, None])
    return combis


def prompt_generation(row, lang):
    prompt = generate_prompt(
        query = row['question'],
        language = lang,
        context = row['context'],
        sociodemographic_info = row['sociodemographic_info'],
        sociodemographic_group = row['sociodemographic_group'],
        explanation = row['info'],
        pros = row['pro'],
        cons = row['contra'],
        stance = row['stance']
    )
    return prompt


for lang in ['de', 'it', 'fr']:
    '''
    # entries per language:
    de: 10118
    it: 333
    fr: 2307
    '''

    # add sociodemographic variance for all fixed issue+stance pairs
    corpus_lang = corpus[corpus['language']==lang]
    combinations = list(chain.from_iterable(corpus_lang.apply(compute_combinations, axis=1).tolist()))
    prompt_df = pd.DataFrame(combinations, columns=['question', 'stance', 'sociodemographic_info', 'sociodemographic_group'])

    '''
    results in (7 (different sociodemographic infos) + 1 (no sociodemographics)) x # entries per language:
    de: 80858 (80944 - 5 nan entries - 81 "Keine Geschlecht" entries)
    it: 2664
    fr: 18405 (18456 - 13 nan entries - 38 "Keine Geschlecht" entries)
    '''

    # remove duplicates
    prompt_df = prompt_df.drop_duplicates()

    '''
    results in # entries per language:
    de: 3092
    it: 1068
    fr: 2247
    '''

    # add explanation variance (for each language, 25 of the 40 issues come with an explanation)
    df_duplicates = prompt_df.copy()
    questions_with_explanation = corpus_lang[corpus_lang['info'].notnull()].drop_duplicates(subset='question')
    rows_to_duplicate = df_duplicates[df_duplicates['question'].isin(questions_with_explanation['question'])]
    rows_to_duplicate = pd.merge(rows_to_duplicate, questions_with_explanation[['question', 'info']], on='question', how='inner')

    prompt_df['info'] = None
    prompt_df = pd.concat([prompt_df, rows_to_duplicate], ignore_index=True)

    '''
    results in # entries per language:
    de: 5089
    it: 1799
    fr: 3770
    '''

    # add pro/con variance (for each language, 16 of the 40 issues come with pro/con examples)
    df_duplicates = prompt_df.copy()
    questions_with_proscons = corpus_lang[corpus_lang['pro'].notnull()].drop_duplicates(subset='question')  # each pro has always a partner contra and vice versa
    rows_to_duplicate = df_duplicates[df_duplicates['question'].isin(questions_with_proscons['question'])]
    rows_to_duplicate = pd.merge(rows_to_duplicate, questions_with_proscons[['question', 'pro', 'contra']], on='question', how='inner')

    prompt_df['pro'], prompt_df['contra'] = None, None
    prompt_df = pd.concat([prompt_df, rows_to_duplicate], ignore_index=True)

    '''
    results in # entries per language:
    de: 7637
    it: 2703
    fr: 5742
    '''

    # add context variance
    rows_to_duplicate = prompt_df.copy()
    prompt_df['context'], rows_to_duplicate['context'] = False, True
    prompt_df = pd.concat([prompt_df, rows_to_duplicate], ignore_index=True)

    '''
    results in # entries per language is doubled:
    de: 15274
    it: 5406
    fr: 11484
    '''

    # prompt generation
    prompt_df['prompt'] = prompt_df.apply(prompt_generation, axis=1, args=(lang,))

    # create unique id
    prompt_df['id'] = prompt_df.index
    prompt_df['id'] = prompt_df['id'].apply(lambda x: lang + "_" + str(x))

    # save generated prompts
    prompt_df.to_csv('prompts_'+lang+'.csv', 
                     columns=[
                         'id', 'prompt', 'question', 'stance', 'sociodemographic_info',
                         'sociodemographic_group', 'info', 'pro', 'contra', 'context'], 
                     index=False)
