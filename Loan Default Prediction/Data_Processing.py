import pandas as pd

# Data Processing functions
# global dictionaries

state_to_region = {
    'CA': 'West', 'OR': 'West', 'UT': 'West', 'WA': 'West', 'CO': 'West',
    'NV': 'West', 'AK': 'West', 'MT': 'West', 'HI': 'West', 'WY': 'West', 'ID': 'West',
    'AZ': 'SouthWest', 'TX': 'SouthWest', 'NM': 'SouthWest', 'OK': 'SouthWest',
    'GA': 'SouthEast', 'NC': 'SouthEast', 'VA': 'SouthEast', 'FL': 'SouthEast', 'KY': 'SouthEast',
    'SC': 'SouthEast', 'LA': 'SouthEast', 'AL': 'SouthEast', 'WV': 'SouthEast', 'DC': 'SouthEast',
    'AR': 'SouthEast', 'DE': 'SouthEast', 'MS': 'SouthEast', 'TN': 'SouthEast',
    'IL': 'MidWest', 'MO': 'MidWest', 'MN': 'MidWest', 'OH': 'MidWest', 'WI': 'MidWest',
    'KS': 'MidWest', 'MI': 'MidWest', 'SD': 'MidWest', 'IA': 'MidWest', 'NE': 'MidWest',
    'IN': 'MidWest', 'ND': 'MidWest',
    'CT': 'NorthEast', 'NY': 'NorthEast', 'PA': 'NorthEast', 'NJ': 'NorthEast', 'RI': 'NorthEast',
    'MA': 'NorthEast', 'MD': 'NorthEast', 'VT': 'NorthEast', 'NH': 'NorthEast', 'ME': 'NorthEast'
}

emp_length_to_int = {'10+ years': 10,
    '9 years': 9,
    '8 years': 8,
    '7 years': 7,
    '6 years': 6,
    '5 years': 5,
    '4 years': 4,
    '3 years': 3,
    '2 years': 2,
    '1 year': 1,
    '< 1 year': 0.5,
    'n/a': 0
}

sub_grade_to_risk = {}
letters = "ABCDEFG"
numbers ="12345"
i = 1
for letter in letters:
    for number in numbers:
        sub_grade_to_risk[letter+number] = i
        i += 1

def compute_months(df:pd.DataFrame, date1:str, date2:str) -> pd.Series:
    d1 = pd.to_datetime(df[date1], format = "%b-%Y", errors="coerce")
    d2 = pd.to_datetime(df[date2], format = "%b-%Y", errors = "coerce")
    diff = (d1.dt.year - d2.dt.year) * 12 + (d1.dt.month - d2.dt.month)
    return diff

def remove_columns(df:pd.DataFrame, thresh:float = 0.7) -> pd.DataFrame:
    if not (0.0 <= thresh <= 1.0):
        raise ValueError("Invalid threshold. Must be between 0 and 1")
    missing_cols = list(df.columns[df.isna().mean() >= 0.7])
    lf_columns = ['last_pymnt_d','next_pymnt_d', 'hardship_flag', 'disbursement_method',
                 'debt_settlement_flag', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
                 'collection_recovery_fee', 'last_pymnt_amnt', 'last_credit_pull_d',
                 'last_fico_range_high', 'last_fico_range_low']
    extra_columns_to_remove = [col for col in df.columns if col.startswith("hardship") or "settlement" in col] + \
            ["id", "emp_title", "url", "title", "zip_code", "policy_code", "earliest_cr_line", "desc", "initial_list_status"]
    df = df.drop(columns = missing_cols + lf_columns + extra_columns_to_remove)
    return df

def map_categorical_variables(df:pd.DataFrame, state_to_region_map:dict,
                             sub_grade_to_risk_map:dict, emp_lenght_to_int_map:dict) -> pd.DataFrame:
    
    df["state"] = df["addr_state"].map(state_to_region_map)
    df["risk_grade"] = df["sub_grade"].map(sub_grade_to_risk_map)
    df["emp_length_int"] = df["emp_length"].map(emp_lenght_to_int_map)
    # Drop columns after they have been processed
    df = df.drop(columns=["grade", "sub_grade", "emp_length", "addr_state"])
    return df

def modify_target_binary(df:pd.DataFrame, target:str) -> pd.Series:
    
    good_statuses = ["Fully Paid",
                   'Does not meet the credit policy. Status:Fully Paid']
    df[target] = df[target].isin(good_statuses).astype(int)
    return df

def imputation(df:pd.DataFrame):
    median_cols = ["risk_grade", "mo_sin_old_il_acct", "bc_util", "int_rate", "installement",
                              "fico_range_low", "fico_range_high", "mths_since_last_delinq", "open_acc",
                              "total_acc", "open_act_il", "il_util", "open_rv_24m", "acc_open_past_24mths",
                              "mo_sin_old_rev_tl_op", "mths_since_recent_inq", "num_actv_rev_tl", "num_bc_sats",
                              "num_bc_tl", "num_sats"]
    
    mean_cols =  ["revol_util", "all_util", "mths_since_recent_revol_delinq", "months_since_earliest_cr"]
    for col in df.columns:
        if col in median_cols:
            df[col] = df[col].fillna(df[col].median())
        elif col in mean_cols:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode())
    return df

def one_hot_encoding(df:pd.DataFrame):
    
    categorical_cols = df.select_dtypes("object").columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    return df_encoded

# Combine everything into one function to process features dataframe
def preprocessing(df:pd.DataFrame):
    df["months_since_earliest_cr"] = compute_months(df, "issue_d", "earliest_cr_line")
    df = map_categorical_variables(df, state_to_region, sub_grade_to_risk, emp_length_to_int)
    df = remove_columns(df)
    df = imputation(df)
    df = one_hot_encoding(df)
    df = df.fillna(0)
    
    return df
