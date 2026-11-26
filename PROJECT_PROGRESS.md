# MP076 Project Progress

## Completed and verified
- [x] Git branch confirmed as `manas`
- [x] Data cleaning and preprocessing implemented
- [x] Data validation and leakage/quality checks implemented
- [x] Data cleaning tests passed
- [x] Feature engineering implemented
- [x] Dummy, Ridge, Random Forest, and HistGradientBoosting models compared
- [x] Time-series cross-validation implemented
- [x] Later-period adaptation evaluated
- [x] MLflow experiment tracking implemented
- [x] Final model artifact generated
- [x] Holdout predictions generated (250 records)
- [x] Drift monitoring and drift report implemented
- [x] FastAPI health, prediction, batch prediction and metrics endpoints implemented
- [x] API validation and test coverage implemented
- [x] Docker image built successfully
- [x] Docker health endpoint tested successfully
- [x] Docker single prediction tested successfully
- [x] Docker batch prediction tested successfully
- [x] Complete pytest suite passed (15 tests)

## Implemented, but pending final verification
- [ ] Run complete DVC pipeline end-to-end
- [ ] Verify GitHub Actions workflow
- [ ] Verify Kubernetes deployment
- [ ] Final README and repository audit
- [ ] Final commit and push to `origin/manas`

## Known limitation
- MLflow model registry registration was unsuccessful because of model serialization/trust restrictions. Experiment tracking and metrics are available, but registry registration is not confirmed.
- Pytest reports one Starlette/httpx deprecation warning; tests still pass.
