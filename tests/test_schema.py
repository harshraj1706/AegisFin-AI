from app.mapper import request_to_dataframe
from app.schemas import PredictionRequest

def test_rich_schema_mapping():
    req = PredictionRequest(
        application={
            "contract_type": "Cash loans",
            "credit_amount": 800000,
            "annual_income": 450000,
            "gender": "M",
            "owns_car": True,
            "owns_realty": True,
            "age_years": 32,
            "employment_years": 6,
            "own_car_age_years": 4,
        },
        enrichment={
            "ext_source_3": 0.71,
            "region_rating_client_w_city": 2,
            "flag_document_3": True,
        },
    )
    df = request_to_dataframe(req)
    assert df.loc[0, "AMT_CREDIT"] == 800000
    assert df.loc[0, "AMT_INCOME_TOTAL"] == 450000
    assert df.loc[0, "FLAG_OWN_CAR"] == "Y"
    assert df.loc[0, "FLAG_OWN_REALTY"] == "Y"
    assert df.loc[0, "DAYS_BIRTH"] < 0
    assert df.loc[0, "DAYS_EMPLOYED"] < 0
    assert df.loc[0, "EXT_SOURCE_3"] == 0.71
    assert df.loc[0, "FLAG_DOCUMENT_3"] == 1
