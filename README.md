# AutoClaim-AI

## 1. Cai dat
pip install -r requirements.txt

## 2. Test
pytest ai/tests/test_pipeline_unit.py -v

## 3. Chay API
uvicorn ai.api.main:app --host 0.0.0.0 --port 8000
