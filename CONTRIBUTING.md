# Contributing to benchmarking

Thank you for your interest in this project.

If you are interested in making a contribution, then please familiarize
yourself with our technical contribution standards as set forth in this guide.

Next, please request appropriate write permissions by [opening an
issue](https://github.com/tenstorrent/benchmarking/issues/new/choose) for
GitHub permissions.

All contributions require:

- an issue
  - Your issue should be filed under an appropriate project. Please file a
    feature support request or bug report under Issues to get help with finding
    an appropriate project to get a maintainer's attention.
- a pull request (PR).
  - Your PR must be approved by appropriate reviewers.

## Setting up environment

Install all dependencies from [requriements-dev.txt](requirements-dev.txt) and install pre-commit hooks in a Python environment with PyBUDA installed.

```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
pre-commit install
```

## Developing model_demos

### Adding models

Contribute to benchmarking by include Python script files under the respective model type directories in `benchmark/models`. If it's a new model architecture, please create a directory for that model. The script should be self-contained and include pre/post-processing steps.

```bash
benchmarking/
├── models/
│ ├── resnet/
│ │ └── resnet.py
│ ├── new_model_arch/
│ │ └── new_model.py
```

If external dependencies are required, please add the dependencies to the [requriements.txt](requirements.txt) file.

To add a model, add a new file to `models/` directory (or add to an existing one) and create a function with the name of your model, and decorate it with
`@benchmark_model` decorator. If your model supports configurations, add `configs=[....]` parameter to the decorator to define the legal configs. For example:

```python
@benchmark_model(configs=["tiny", "base", "large"])
def bert(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):
```

`config` is an optional parameter to your model, but should be there if you've defined legal configs in the decorator.

Finally, import your newly defined model to the `benchmark.py` runtime:

```python
# Models
import benchmark.models.bert.bert
```

### Cleaning the dev environment

`make clean` and `make clean_tt` clears out model and build artifacts. Please make sure no artifacts are being pushed.

### Running pre-commit hooks

You must run hooks before you commit something.

To manually run the style formatting, run:

```bash
make style
```

## Contribution standards

### Code reviews

- A PR must be opened for any code change with the following criteria:
  - Be approved by a maintaining team member and any code owners whose modules are relevant for the PR.
  - Run pre-commit hooks.
  - Pass automated GitHub Actions workflow tests.
  - Pass any acceptance criteria mandated in the original issue.
  - Pass any testing criteria mandated by code owners whose modules are relevant for the PR.

For more information on the GitHub Actions and Pull Request Workflow, please see the [GitHub Actions and Pull Request Workflow section](#github-actions-and-pull-request-workflow) within this document.

### GitHub Actions and Pull Request Workflow

#### Automated Linting with GitHub Actions

Linting, styling, and cleaning checks are automatically performed on pull requests using GitHub Actions. This ensures that contributed code meets standard Python coding standards before it's merged into the main branch.

1. **Pull Request Process**: When you open a pull request, GitHub Actions will automatically trigger linting, styling, and cleaning checks on the changes made within the `benchmarking` directory.
2. **Approval Requirement**: In order to merge a pull request, it must pass the GitHub Actions workflow test. This ensures that all contributions adhere to our coding standards and maintain consistency throughout our `benchmarking` repository.
3. **Interpreting Results**: If linting fails on your pull request, review the output to identify and fix any issues. You'll need to address these issues before the pull request can be approved and merged. In case of repeated failures or failures within the GitHub Actions workflow files, please reach out to one of the repository maintainers from the [Maintainers.md](MAINTAINERS.md).

#### Automated Commit by GitHub Actions

The GitHub Actions workflow also automatically makes a commit with the message `*** AUTOMATED COMMIT | Applied Code Formatting and Cleanup ✨ 🍰 ✨***` authored by `[anirudTT]` when it performs code formatting and cleanup. If you open a pull request and subsequently push more changes, we suggest **rebasing or pulling again** from the pull request branch before pushing your changes again to avoid conflicts.

#### SPDX License Compliance Check

Our project enforces SPDX license compliance for all files through automated checks integrated within our GitHub Actions workflows. These workflows, `check-license.yml` and `license-checker.yml`, automatically verify the presence and accuracy of SPDX license headers and the specified license year, correct company mentions within license headers in files included in pull requests.

- **License Validation**: The `check-license.yml`  workflow checks for the correct SPDX license year and the proper mention of the company name, "Tenstorrent AI ULC", in newly changed files. If files are found lacking either the correct year, the correct license identifier, or the proper company mention, the workflow comments on the PR with details about the missing elements and fails to highlight the need for compliance. This ensures that all changes adhere to licensing standards and maintain corporate copyright claims properly.

- **License Header Checks**: Similarly, the `license-checker.yml` workflow assesses if files in the PR contain valid SPDX license headers as per the configurations specified in `check_copyright_config.yaml`. Non-compliant files are listed in an automated comment on the PR, urging contributors to update their files accordingly.

- **Action on Non-Compliance**: PRs flagged for non-compliance with SPDX license requirements will not pass the automated checks, preventing them from being merged. Contributors are encouraged to follow the detailed instructions in the automated comments to rectify any issues and achieve compliance.


##### Actions to Take if Your PR Fails SPDX License Compliance Checks

If your pull request fails due to non-compliance with SPDX license requirements, it's crucial to address these issues promptly to ensure your contributions can be merged into the project. Here's what you need to do:

###### Review Automated Comments

1. **Identify Issues**: Carefully review the automated comments left by the `check-license.yml` and `license-checker.yml` workflows on your pull request. These comments will specify which files are non-compliant and what specific compliance issues need to be addressed (e.g., incorrect license year, missing SPDX license header).

2. **Understand Required Changes**: Each comment will provide detailed instructions on how to rectify the compliance issues. This may involve adding or updating the SPDX license headers in the specified files or correcting the license year or the company name within the license.

###### Correcting Compliance Issues

1. **Update Your Files**: Based on the guidance provided in the automated comments, make the necessary changes to your files.

2. **Verify Changes Locally**: Before pushing your changes, it's a good practice to locally review the files you've updated to ensure they now comply with the project's SPDX license requirements.

3. **Push Your Updates**: Once you've made the necessary corrections, commit your changes and push them to your branch associated with the pull request. GitHub Actions will automatically re-run the compliance checks on the updated pull request.

###### After Correcting Issues

- **Automated Re-check**: After you push your updates, the GitHub Actions workflows will automatically re-run the SPDX license compliance checks on the modified pull request. If all issues have been correctly addressed, the previously failed checks should now pass.

- **Seek Further Assistance**: If you've made the recommended changes but your pull request still fails the compliance checks, or if you need clarification on the required actions, don't hesitate to ask for help. You can comment on your pull request tagging a maintainer, or refer to the [Maintainers.md](Maintainers.md) document to contact maintainers directly.

