\# EconoCausal



\*\*Causal ML for personalized discounts.\*\* Standard ML predicts who will buy. EconoCausal estimates who will buy

\*because of\* a discount (the Individual Treatment Effect) using Double Machine Learning, then allocates a fixed budget to the

customers where discounts pay off.



\## Quick start



```bash

\# Backend (Python 3.10+)

cd backend

pip install -r requirements.txt

python -m pytest                     # automated tests

python scripts/run\_pipeline.py       # end-to-end demo in the terminal

uvicorn econocausal.api:app --reload # API on http://localhost:8000  (docs at /docs)



\# Dashboard (Node 18+), in a second terminal

cd frontend

npm install

npm run dev                          # http://localhost:5173

```



In the dashboard: \*\*Use demo retail data -> Train causal model -> Allocation -> set budget\*\*.



\## What the demo shows

With 6,000 historical customers (confounded: past discounts went to low-engagement users) and a $5,000 budget.

All results use synthetic data where the true effect is known.



| | Naive regression | Double ML | Ground truth |

|---|---|---|---|

| Avg. lift in buy probability per $1 | +0.40 pts | +0.85 pts | +0.94 pts |



| Strategy (same budget) | True profit gain |

|---|---|

| Random blanket discount | about -$270 |

| Predictive targeting (lowest buy probability) | about +$1,520 |

| \*\*EconoCausal\*\* | \*\*about +$2,930\*\* |



Exact numbers vary slightly by library versions. The causal model also stops spending once extra discounts would lose money.



\## Modules

| Project brief | Code |

|---|---|

| Causal Inference Engine (EconML, DoWhy) | `dag.py`, `dml.py`, `refutation.py` |

| Propensity Score Matching | `propensity.py` |

| Optimization Solver (SciPy) | `optimizer.py`, `strategies.py` |

| Experimentation UI (React \& Plotly) | `frontend/` |

| REST API and drift detector | `api.py`, `drift.py` |



\## API

`POST /data/demo` | `POST /data/upload` | `POST /train` | `GET /uplift` | `GET /audit` | `POST /allocate` |

`GET /allocate/download` | `GET /users/{id}/ite` | `POST /drift/demo` | `POST /drift/upload` | `GET /dag`



\## Using your own data

Upload a CSV with: `user\_id, age, income\_k, tenure\_months, past\_purchases, engagement, cart\_value, discount, purchased, margin`

(`GET /data/template` gives a template). See `docs/ARCHITECTURE.md` for method details and limitations.



\## Docker (backend only)

```bash

docker build -t econocausal .

docker run -p 8000:8000 econocausal

```

