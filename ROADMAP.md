# Roadmap

This roadmap documents action items such as features or bugs to be developed/fixed.

_Updated: 31 Mar 2024, 23:10 GMT_

## MLOps Test

| Status | Item                                                                                                                                                                                                                                                                               |
| :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   ✔    | Create README and ROADMAP                                                                                                                                                                                                                                                          |
|   ✔    | Write simple logistic regression classifier for bank customer churn                                                                                                                                                                                                                |
|   ✔    | Write Dockerfile                                                                                                                                                                                                                                                                   |
|   ✔    | Write GitHub Actions workflow                                                                                                                                                                                                                                                      |
|   ✔    | Write _serverless.yml_ for Serverless Framework Deployment                                                                                                                                                                                                                         |
|   ✔    | Configure AWS S3 buckets for upload and download of trained models and metadata                                                                                                                                                                                                    |
|   ✔    | Implement switch_first=True whether to begin with a Switch layer                                                                                                                                                                                                                   |
|   ✔    | Write pytests and integrate into workflow                                                                                                                                                                                                                                          |
|   ❌   | Decide and integrate more complex / interesting ML models/applications. This may potentially be carried out in a separate repository. Current ideas include conformal prediction on top of NLP tasks with BERT or autonomous driving tasks like traffic light detection, and more. |
|   ❌   | Integrate call to `train_handle` into _serverless.yml_ or _Dockerfile_ to re-train model upon push                                                                                                                                                                                 |
