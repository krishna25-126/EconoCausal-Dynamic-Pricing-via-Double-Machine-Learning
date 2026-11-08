\# Architecture



```

React + Plotly dashboard  --HTTP/JSON-->  FastAPI (api.py)

&#x20;                                             |

&#x20;                                       service.py  (session state, orchestration)

&#x20;       +---------------+--------------+------+-------+--------------+------------+

&#x20;    data.py        dag.py +        dml.py        optimizer.py   drift.py     summary.py

&#x20; (synthetic      refutation.py   (EconML        strategies.py  (KS + PSI)   (plain-

&#x20;  retail +       propensity.py   CausalForest-   (SciPy MILP)                language)

&#x20;  ground truth)  (DoWhy, PSM)    DML)           evaluation.py (Qini)

```



\## Data flow

1\. \*\*Data.\*\* Historical campaign rows (features, discount, purchase, margin). The demo data has a known true effect.

2\. \*\*Identification (DoWhy).\*\* The DAG says every feature confounds discount and purchase, so backdoor adjustment on those

&#x20;  features identifies the causal effect.

3\. \*\*Estimation (EconML).\*\* `CausalForestDML` with LightGBM nuisance models. Cross-fitting predicts outcome and discount

&#x20;  from features; the forest regresses residual outcome on residual discount, giving a per-customer effect `tau(x)` (change

&#x20;  in purchase probability per $1).

4\. \*\*Evaluation.\*\* Qini curve and decile uplift on a randomized hold-out; on synthetic data also correlation with true effects.

5\. \*\*Audit.\*\* DoWhy refuters: placebo treatment, random common cause, data subset.

6\. \*\*Prescription.\*\* For each customer and discount level d: `p(d) = clip(p0 + tau\*d)`, `cost = p(d)\*d`,

&#x20;  `gain = p(d)\*(margin - d) - p0\*margin`. A mixed-integer program picks one level per customer to maximize total gain

&#x20;  subject to expected spend <= budget.

7\. \*\*Monitoring.\*\* Kolmogorov-Smirnov test and Population Stability Index compare new customers to the training data.



\## Design decisions

\- \*\*Budget is a cap.\*\* If another discount would lose money, the optimizer leaves budget unspent.

\- \*\*Baseline probability\*\* `p0(x)` is recovered from the DML identity `E\[Y|X] = p0(X) + tau(X) \* E\[T|X]`.

\- \*\*Honest evaluation.\*\* Uplift is judged on randomized data, never on the confounded training set.



\## Limitations

\- Demo results use synthetic data; real data needs overlap (customers with and without discounts across all feature ranges).

\- DML assumes no unmeasured confounders. The refutation tests help but cannot prove this.

\- The effect is modelled as linear in the discount amount; real price response may saturate.

