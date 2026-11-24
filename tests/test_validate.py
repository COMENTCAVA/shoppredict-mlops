
import pandas as pd
import pytest

from ml.validate import validate


def make_dataset(month):
    return pd.DataFrame(
        {
            "Administrative": [2],
            "Administrative_Duration": [30.0],
            "Informational": [1],
            "Informational_Duration": [10.0],
            "ProductRelated": [15],
            "ProductRelated_Duration": [250.0],
            "BounceRates": [0.02],
            "ExitRates": [0.04],
            "PageValues": [5.0],
            "SpecialDay": [0.0],
            "OperatingSystems": [2],
            "Browser": [1],
            "Region": [3],
            "TrafficType": [2],
            "VisitorType": ["Returning_Visitor"],
            "Weekend": [0],
            "Revenue": [1],
            "Month": [month],
        }
    )


def test_valid_data():
    validate(make_dataset("Feb"), make_dataset("Dec"))


def test_missing_column():
    train = make_dataset("Feb").drop(columns=["PageValues"])

    with pytest.raises(ValueError, match="missing columns"):
        validate(train, make_dataset("Dec"))


@pytest.mark.parametrize(
    "column, value",
    [
        ("Revenue", 2),
        ("BounceRates", 1.5),
        ("ExitRates", -0.1),
        ("SpecialDay", 2.0),
        ("PageValues", -5.0),
        ("Administrative", -1),
        ("OperatingSystems", 0),
        ("Browser", 0),
        ("Region", 10),
        ("TrafficType", 0),
        ("Weekend", 2),
        ("VisitorType", "Unknown"),
    ],
)
def test_invalid_values(column, value):
    train = make_dataset("Feb")
    train[column] = value

    with pytest.raises(ValueError, match=column):
        validate(train, make_dataset("Dec"))


def test_missing_value():
    train = make_dataset("Feb")
    train["PageValues"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate(train, make_dataset("Dec"))


def test_month_overlap():
    with pytest.raises(ValueError, match="month overlap"):
        validate(make_dataset("Feb"), make_dataset("Feb"))
