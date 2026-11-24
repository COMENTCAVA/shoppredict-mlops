
from pandas.api.types import is_numeric_dtype

FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

TARGET = "Revenue"

NON_NEGATIVE = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "PageValues",
]

RATE_COLUMNS = [
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]

POSITIVE_COLUMNS = [
    "OperatingSystems",
    "Browser",
    "TrafficType",
]

INTEGER_COLUMNS = [
    "Administrative",
    "Informational",
    "ProductRelated",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

VISITOR_TYPES = [
    "Returning_Visitor",
    "New_Visitor",
    "Other",
]


def validate(train_df, eval_df):
    required = FEATURES + [TARGET, "Month"]

    for name, df in [("train", train_df), ("eval", eval_df)]:
        missing = set(required) - set(df.columns)

        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")

        if df.empty:
            raise ValueError(f"{name}: empty dataset")

        if df[required].isna().any().any():
            raise ValueError(f"{name}: missing values")

        if not df[TARGET].isin([0, 1]).all():
            raise ValueError(f"{name}: invalid Revenue values")

        if not df["Weekend"].isin([0, 1]).all():
            raise ValueError(f"{name}: invalid Weekend values")

        if not df["VisitorType"].isin(VISITOR_TYPES).all():
            raise ValueError(f"{name}: invalid VisitorType")

        numeric_columns = (
            NON_NEGATIVE + RATE_COLUMNS + POSITIVE_COLUMNS + ["Region"]
        )

        for column in numeric_columns:
            if not is_numeric_dtype(df[column]):
                raise ValueError(f"{name}: {column} must be numeric")

            if df[column].isin([float("inf"), float("-inf")]).any():
                raise ValueError(f"{name}: invalid {column}")

        for column in INTEGER_COLUMNS:
            if not (df[column] % 1 == 0).all():
                raise ValueError(f"{name}: {column} must be an integer")

        for column in NON_NEGATIVE:
            if (df[column] < 0).any():
                raise ValueError(f"{name}: invalid {column}")

        for column in RATE_COLUMNS:
            if not df[column].between(0, 1).all():
                raise ValueError(f"{name}: invalid {column}")

        for column in POSITIVE_COLUMNS:
            if (df[column] < 1).any():
                raise ValueError(f"{name}: invalid {column}")

        if not df["Region"].between(1, 9).all():
            raise ValueError(f"{name}: invalid Region")

    overlap = set(train_df["Month"]) & set(eval_df["Month"])

    if overlap:
        raise ValueError(f"Train/eval month overlap: {sorted(overlap)}")
