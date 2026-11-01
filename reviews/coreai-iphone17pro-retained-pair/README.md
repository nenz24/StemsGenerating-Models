# iPhone retained pair: dual-reference PASS, first-use cost amortized

[report.txt](report.txt) transcribes the user-pasted device log;
[report.json](report.json) retains all metrics and scope. One model and one
function were opened/bound for four calls with the same fixed inputs.

| Run | Backend-await | Read-back/checks | Both references |
| --- | --- | --- | --- |
| 1 | 1042.534584 ms | 1.094333 ms | PASS |
| 2 | 8.438375 ms | 0.811625 ms | PASS |
| 3 | 8.524209 ms | 0.773875 ms | PASS |
| 4 | 8.600958 ms | 0.763167 ms | PASS |

Repeated mean backend-await is **8.521180666666666 ms**; the first/repeated
component ratio is **122.34625984145703**. All eight comparisons pass. Original
FP32 normalized RMSE is 7.176440958068637e-7 and storage16 normalized RMSE is
6.879016138198428e-7 on each call. Absolute errors above 1e-4 still meet the
unchanged combined absolute-plus-relative bound.

The result supports large first-use-associated work that can be amortized in
this pair/function scope. It does not identify an internal initialization or
paging mechanism, guarantee a cold first call, or establish a 122× loop speedup.
The entire probe takes 1274.0055419999999 ms including setup/checks/reporting.

Sampled process baseline is 58.73615264892578 MiB and maximum is
58.89240264892578 MiB. These are not total model/device RAM or proof that all
14 handles will fit. Residency and accelerator placement remain unmeasured
under requested CPU-only policy.

This completes the **first of two planned main stage-1 experiments** for this
synthetic case. [The second experiment](../../docs/coreai-retained-loop.md) retains
all 14 handles across the full four-step loop, loading lazily with footprint
checks and unchanged references. [The full-loop experiment now also passes](../coreai-iphone17pro-retained-loop/README.md),
completing both planned main prototype stage-1 tests. No native model export is
needed. Recording conditioning and music remain later work.
