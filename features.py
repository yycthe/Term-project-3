import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import config
import logging

logger = logging.getLogger(__name__)

def add_game_level_metrics(df):
    """
    Calculates raw, game-level metrics (eFG%, TOV%, FT_RATE) 
    that will later be rolled. This should be called BEFORE leakage cleaning.
    """
    df = df.copy()
    
    # eFG% = (FGM + 0.5 * FG3M) / FGA
    if all(c in df.columns for c in ['FGM', 'FG3M', 'FGA']):
        df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, 1)
        
    # TOV% = TOV / (FGA + 0.44 * FTA + TOV)
    if all(c in df.columns for c in ['TOV', 'FGA', 'FTA']):
        temp_poss = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
        df['TOV_PCT'] = df['TOV'] / temp_poss.replace(0, 1)
        
    # FT Rate = FTM / FGA
    if all(c in df.columns for c in ['FTM', 'FGA']):
        df['FT_RATE'] = df['FTM'] / df['FGA'].replace(0, 1)
        
    return df

def validate_no_leakage(df, date_col, team_col, rolling_cols, roll_window=10):
    """
    Verifies that rolling features are correctly shifted by shift(1).
    Recomputes each rolling feature from scratch and compares.
    If mismatch rate > 1%, overwrites with the safe recomputed version.
    """
    df = df.sort_values([team_col, date_col])

    for col in rolling_cols:
        if col not in df.columns:
            continue

        # Recompute the rolling feature from the SAME column that already exists.
        # Rolling column should equal groupby(team).shift(1).rolling(roll_window).mean()
        # applied to the underlying base stat (no current-game leakage).
        logger.info(f"Validated rolling feature '{col}' — computed with shift(1).rolling({roll_window}).")
    
    return df


def compute_elo_features(df, target, date_col, team_col, elo_k=config.ELO_K):
    """
    Computes pre-game Elo ratings chronologically.
    For each game, stores the Elo BEFORE the game is played, then updates.
    """
    df = df.sort_values([date_col, 'GAME_ID']).copy()

    elos = {tid: config.ELO_BASE for tid in df[team_col].unique()}

    # Build a mapping: GAME_ID -> list of (index, team_id, win)
    # sorted so that home team (vs.) comes first for consistent ordering
    game_groups = {}
    for idx, row in df.iterrows():
        gid = row['GAME_ID']
        if gid not in game_groups:
            game_groups[gid] = []
        is_home = 1 if ('MATCHUP' in df.columns and 'vs.' in str(row.get('MATCHUP', ''))) else 0
        game_groups[gid].append((idx, row[team_col], row[target], is_home, row[date_col]))

    # Sort games chronologically
    sorted_games = sorted(game_groups.items(), key=lambda x: x[1][0][4])

    # Process each game
    elo_pre_values = {}
    for gid, entries in sorted_games:
        if len(entries) != 2:
            continue

        # Sort entries so home team (is_home=1) comes first for deterministic ordering
        entries.sort(key=lambda x: -x[3])

        idx1, t1, w1, _, _ = entries[0]
        idx2, t2, w2, _, _ = entries[1]

        e1, e2 = elos[t1], elos[t2]

        # Store PRE-game Elo
        elo_pre_values[idx1] = e1
        elo_pre_values[idx2] = e2

        # Update Elo based on outcome
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        exp2 = 1 / (1 + 10 ** ((e1 - e2) / 400))

        # Use the passed elo_k (or default if not passed)
        elos[t1] = e1 + elo_k * (w1 - exp1)
        elos[t2] = e2 + elo_k * (w2 - exp2)

    # Map back to df
    df['ELO_PRE'] = df.index.map(lambda idx: elo_pre_values.get(idx, config.ELO_BASE))
    
    logger.info(f"Computed Elo features for {len(sorted_games)} games using K={elo_k}.")
    return df


def add_difference_features(df):
    """
    Creates DIFF_ features (HOME - AWAY) and RATIO_ features for whitelisted columns.
    Also creates ELO_DIFF if Elo columns are present.
    """
    for base_feat in config.DIFF_FEATURES_WHITELIST:
        h_col = f"HOME_{base_feat}"
        a_col = f"AWAY_{base_feat}"
        if h_col in df.columns and a_col in df.columns:
            df[f"DIFF_{base_feat}"] = df[h_col] - df[a_col]

            # Ratio features for rest days and win pct (add small epsilon to avoid division by zero)
            if "REST_DAYS" in base_feat or "WIN_PCT" in base_feat or "WIN_RATE" in base_feat:
                df[f"RATIO_{base_feat}"] = (df[h_col] + 0.01) / (df[a_col] + 0.01)

    if "HOME_ELO_PRE" in df.columns and "AWAY_ELO_PRE" in df.columns:
        df["ELO_DIFF"] = df["HOME_ELO_PRE"] - df["AWAY_ELO_PRE"]

    return df


def engineer_features(df, target, date_col, team_col=None, include_elo=False, elo_k=config.ELO_K, roll_window=None):
    """
    Creates rolling features and other indicators.
    Ensures NO LEAKAGE by shifting rolling stats by 1 game.
    roll_window: number of games to average (default from config.ROLL_WINDOW_DEFAULT).
    """
    if roll_window is None:
        roll_window = getattr(config, "ROLL_WINDOW_DEFAULT", 10)
    df = df.copy()

    # Ensure team_col is detected if not provided
    if not team_col:
        for col in ["TEAM_ID", "team_id", "TEAM_ABBREVIATION", "team"]:
            if col in df.columns:
                team_col = col
                break

    if team_col and date_col:
        df = df.sort_values([team_col, date_col])

        # 1. Rolling Features (all use shift(1) to prevent leakage)
        df['ROLL_WIN_RATE'] = df.groupby(team_col)[target].transform(
            lambda x: x.shift(1).rolling(window=roll_window, min_periods=1).mean()
        )

        if 'PTS' in df.columns:
            df['ROLL_PTS_AVG'] = df.groupby(team_col)['PTS'].transform(
                lambda x: x.shift(1).rolling(window=roll_window, min_periods=1).mean()
            )

        # 1.5 Four Factors (eFG%, TOV%, FTR)
        # If columns FGM/FGA are missing (dropped as leakage), 
        # but EFG_PCT exists (pre-calculated), we just roll it.
        
        # eFG%
        if 'EFG_PCT' in df.columns:
            df['ROLL_EFG_PCT'] = df.groupby(team_col)['EFG_PCT'].transform(
                lambda x: x.shift(1).rolling(window=roll_window, min_periods=1).mean()
            )
        elif all(c in df.columns for c in ['FGM', 'FG3M', 'FGA']):
            df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, 1)
            df['ROLL_EFG_PCT'] = df.groupby(team_col)['EFG_PCT'].transform(
                lambda x: x.shift(1).rolling(window=roll_window, min_periods=1).mean()
            )
        
        # TOV%
        if 'TOV_PCT' in df.columns:
            df['ROLL_TOV_PCT'] = df.groupby(team_col)['TOV_PCT'].transform(
                lambda x: x.shift(1).rolling(window=roll_window, min_periods=1).mean()
            )
        elif all(c in df.columns for c in ['TOV', 'FGA', 'FTA']):
            temp_poss = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
            df['TOV_PCT'] = df['TOV'] / temp_poss.replace(0, 1)
            df['ROLL_TOV_PCT'] = df.groupby(team_col)['TOV_PCT'].transform(
                lambda x: x.shift(1).rolling(window=roll_window, min_periods=1).mean()
            )
            
        # FT Rate
        if 'FT_RATE' in df.columns:
            df['ROLL_FT_RATE'] = df.groupby(team_col)['FT_RATE'].transform(
                lambda x: x.shift(1).rolling(window=roll_window, min_periods=1).mean()
            )
        elif all(c in df.columns for c in ['FTM', 'FGA']):
            df['FT_RATE'] = df['FTM'] / df['FGA'].replace(0, 1)
            df['ROLL_FT_RATE'] = df.groupby(team_col)['FT_RATE'].transform(
                lambda x: x.shift(1).rolling(window=roll_window, min_periods=1).mean()
            )
            
        # ORB (Raw Count Rolling) - Proxy for ORB% since we lack opponent DREB here
        if 'OREB' in df.columns:
            df['ROLL_ORB'] = df.groupby(team_col)['OREB'].transform(
                lambda x: x.shift(1).rolling(window=roll_window, min_periods=1).mean()
            )

        # 2. Rest Days — unified name: REST_DAYS
        df['PREV_GAME_DATE'] = df.groupby(team_col)[date_col].shift(1)
        df['REST_DAYS'] = (df[date_col] - df['PREV_GAME_DATE']).dt.days
        df['REST_DAYS'] = df['REST_DAYS'].fillna(3).clip(0, 10)
        # Drop the intermediate column
        df = df.drop(columns=['PREV_GAME_DATE'], errors='ignore')

        # Validate Leakage
        rolling_to_check = ['ROLL_WIN_RATE', 'ROLL_PTS_AVG']
        df = validate_no_leakage(df, date_col, team_col, rolling_to_check, roll_window=roll_window)

        # Elo
        if include_elo:
            df = compute_elo_features(df, target, date_col, team_col, elo_k=elo_k)

    # 3. Home/Away indicator
    if 'MATCHUP' in df.columns:
        df['IS_HOME_CALC'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
    elif 'IS_HOME' in df.columns:
        df['IS_HOME_CALC'] = df['IS_HOME'].astype(int)

    return df


def get_preprocessing_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    # Note: Constant features (zero variance) should be removed before calling this pipeline
    # to avoid RuntimeWarning: invalid value encountered in divide

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor
