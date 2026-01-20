import pandas as pd
import hashlib
import re

try:
    from data.unified_schema.coaid.deprecate_article_scraping import _claim_norm_hash
except Exception:
    def _claim_norm_hash(title):
        if title is None:
            return None
        s = re.sub(r'\s+', ' ', str(title).strip().lower())
        return hashlib.sha1(s.encode('utf-8')).hexdigest()

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

def cleanup(df:pd.DataFrame, id_col:str = 'id',claim_col:str = 'claim_text', article_col:str = None,
            label_col:str = 'label', dataset:str = None, ) -> pd.DataFrame:
    """

    :param df: cleaned df with most columns in out_cols
    :param id_col: id column name
    :param claim_col: claim text column
    :param label_col: label column name
    :param label_mode: label mode identifier
    :param dataset: dataset name
    :return: cleaned dataset with all necessary columns
    """

    out_cols = [
        'dataset',
        'id',
        'claim_text',
        'article_text',
        'content_status',
        'label_raw',
        'label',
        'label_confidence',
        'label_mode',
        'label_3way',
        'label_bin',
        'source_id',
        'claim_norm_hash',
        'lang',
        'content_char_len'
    ]

    label_mapping_3way = {
        'true' : 2,
        'mixed': 1,
        'false': 0
    }

    label_mapping_bin = {
        'true':1,
        'false' : 0
    }

    source_mapping = {
        'liar':0,
        'coaid':1,
        'fakehealth':2,
        'fakenewsnet':3,
        'hover':4
    }

    temp = df.copy()

    temp['dataset'] = dataset.lower().strip()

    if id_col in temp.columns:
        temp.rename(columns={id_col:'id'},inplace=True)
    else:
        raise KeyError(f'Missing required column: {id_col}.')

    if claim_col in temp.columns:
        temp.rename(columns={claim_col: 'claim_text'}, inplace=True)
    else:
        raise KeyError(f'Missing required column: {claim_col}.')

    if article_col is not None and article_col in temp.columns:
        temp.rename(columns={article_col: 'article_text'}, inplace=True)
    else:
        if 'article_text' not in temp.columns:
            temp['article_text'] = None
        if 'content_status' not in temp.columns:
            temp['content_status'] = 'title_only'
        print('Added/ensured article_text and content_status defaults.')

    if label_col in temp.columns:
        temp.rename(columns={label_col: 'label'}, inplace=True)
        temp['label_raw'] = temp['label']
        temp["label"] = temp["label"].apply(
            lambda x: str(x).lower().strip() if pd.notna(x) else None
        )
    else:
        raise KeyError(f"Missing required column: {label_col}.")

    label_mode = 0 if dataset == 'liar' or dataset == 'fakehealth' else 1
    temp['label_mode'] = label_mode
    if label_mode == 0:
        temp['label_3way'] = temp['label'].map(label_mapping_3way)
        temp['label_bin'] = -100
    else:
        temp['label_3way'] = -100
        temp['label_bin'] = temp['label'].map(label_mapping_bin)

    temp['source_id'] = temp['dataset'].map(source_mapping)

    temp['claim_norm_hash'] = temp['claim_text'].astype(str).apply(_claim_norm_hash)

    if LANGDETECT_AVAILABLE:
        def _safe_detect(x):
            try:
                x = '' if x is None else str(x)
                return detect(x) if x.strip() else None
            except Exception:
                return None
        temp['lang'] = temp['article_text'].apply(_safe_detect)
    else:
        temp['lang'] = None

    if 'content_char_len' not in temp.columns:
        temp['content_char_len'] = temp['article_text'].fillna('').astype(str).apply(len)
    else:
        temp['content_char_len'] = temp["content_char_len"].fillna(0).astype(int)

    return temp[out_cols]
