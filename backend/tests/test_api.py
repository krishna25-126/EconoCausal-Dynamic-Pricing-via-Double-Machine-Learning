import pytest
from fastapi.testclient import TestClient
from econocausal.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_full_flow(client):
    assert client.post("/train").status_code == 400            # no data yet
    assert client.post("/data/demo?n=3000").json()["rows"] == 3000
    r = client.post("/train?trees=100")
    assert r.status_code == 200 and "qini" in r.json()
    a = client.post("/allocate", json={"budget": 3000, "max_rows": 20}).json()
    assert a["expected_spend"] <= 3000 + 1e-6
    assert len(a["table"]) == 20 and "saves" in a["summary"]
    assert client.get("/allocate/download").status_code == 200
    uid = a["table"][0]["user_id"]
    assert len(client.get(f"/users/{uid}/ite").json()["levels"]) == 4
    assert client.get("/users/1/ite").status_code in (200, 404)
    assert client.get("/audit").json()["all_passed"]
    assert client.post("/drift/demo?shift=1.0").json()["status"] == "drift_detected"
    assert client.get("/dag").json()["nodes"]


def test_upload_validation(client):
    r = client.post("/data/upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400 and "Missing required columns" in r.json()["detail"]