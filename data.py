import pandas as pd
import numpy as np
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path=config.DATA_PATH):
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def detect_columns(df):
    """
    Detect target, date, and team columns from the dataframe.
    NOTE: This function also converts the date column to datetime
    and sorts by date as a side effect.
    """
    results = {
        "target": None,
        "date": None,
        "team": None,
        "features": []
    }
    
    # Detect Target
    for cand in config.TARGET_CANDIDATES:
        cols = [c for c in df.columns if c.lower() == cand.lower()]
        if cols:
            results["target"] = cols[0]
            break
            
    if not results["target"]:
        logger.error(f"Target column not found. Columns: {list(df.columns)}")
        return None, None

    # Detect Date
    for cand in config.DATE_CANDIDATES:
        cols = [c for c in df.columns if c.lower() == cand.lower()]
        if cols:
            results["date"] = cols[0]
            break
    
    if results["date"]:
        df[results["date"]] = pd.to_datetime(df[results["date"]])
        df = df.sort_values(results["date"])
    else:
        logger.warning("No date column found. Falling back to index split.")

    return df, results

def leakage_check(df, target, date_col=None):
    """Flags suspicious columns by name patterns and correlation."""
    potential_leakage = []
    
    # Pattern based check
    for col in df.columns:
        if col == target or col == date_col:
            continue
        
        col_lower = col.lower()
        is_leakage = False
        for kw in config.LEAKAGE_KEYWORDS:
            if len(kw) <= 2:
                if col_lower == kw:
                    is_leakage = True
                    break
            else:
                if kw in col_lower:
                    is_leakage = True
                    break
        
        if is_leakage:
            # Exception: if it starts with "ROLL_" or "PREV_" it's likely engineered features
            if not (col_lower.startswith("roll_") or col_lower.startswith("prev_") or col_lower.startswith("rest_") or col_lower in ['efg_pct', 'tov_pct', 'ft_rate']):
                potential_leakage.append(col)
                
    # Correlation based check
    numeric_df = df.select_dtypes(include=[np.number])
    if target in numeric_df.columns:
        corrs = numeric_df.corr()[target].abs().sort_values(ascending=False)
        # Highly correlated columns (excluding target itself)
        high_corr = corrs[corrs > 0.9].index.tolist()
        for col in high_corr:
            col_lower = col.lower()
            # Exception for protected features
            is_protected = (col_lower.startswith("roll_") or 
                          col_lower.startswith("prev_") or 
                          col_lower.startswith("rest_") or 
                          col_lower in ['efg_pct', 'tov_pct', 'ft_rate'])
                          
            if col != target and col not in potential_leakage and not is_protected:
                potential_leakage.append(col)
                
    logger.info(f"Potential leakage columns identified: {potential_leakage}")
    return potential_leakage

def create_matchup_df(df, target, date_col):
    """
    Combines two rows per GAME_ID into one matchup row where:
    - Row with 'vs.' becomes HOME
    - Row with '@' becomes AWAY
    """
    if 'MATCHUP' not in df.columns or 'GAME_ID' not in df.columns:
        logger.warning("GAME_ID or MATCHUP not found. Skipping matchup creation.")
        return df

    # Separate Home and Away
    home_df = df[df['MATCHUP'].astype(str).str.contains('vs.', regex=False, na=False)].copy()
    away_df = df[df['MATCHUP'].astype(str).str.contains('@', regex=False, na=False)].copy()

    # Rename columns with prefix
    exclude_cols = ['GAME_ID', date_col, config.SEASON_COL]
    home_cols = {col: f"HOME_{col}" for col in df.columns if col not in exclude_cols}
    away_cols = {col: f"AWAY_{col}" for col in df.columns if col not in exclude_cols}

    home_df = home_df.rename(columns=home_cols)
    away_df = away_df.rename(columns=away_cols)

    # Merge
    # We include SEASON_COL in the merge key so it doesn't get duplicated or lost
    if config.SEASON_COL in df.columns:
        merge_on = ['GAME_ID', date_col, config.SEASON_COL]
    else:
        merge_on = ['GAME_ID', date_col]
        
    matchup_df = pd.merge(home_df, away_df, on=merge_on)
    
    # Target is HOME_WIN (or whatever the target was in home row)
    if f"HOME_{target}" in matchup_df.columns:
        matchup_df['TARGET'] = matchup_df[f"HOME_{target}"]
    
    logger.info(f"Created matchup-level dataframe with shape {matchup_df.shape}")
    return matchup_df

def get_clean_df(df, target, date_col, drop_leakage=True, matchup=True):
    if drop_leakage:
        leakage_cols = leakage_check(df, target, date_col)
        df = df.drop(columns=leakage_cols)
    
    # Drop rows where target is NaN
    df = df.dropna(subset=[target])
    
    # Convert target to int if binary
    if df[target].dtype == 'object':
        df[target] = df[target].map({'W': 1, 'L': 0, 'win': 1, 'loss': 0, '1': 1, '0': 0})
    
    if matchup:
        df = create_matchup_df(df, target, date_col)
        
    return df
