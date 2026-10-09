# Result visualization

This repository contains Python scripts for analyzing prediction results and creating figures and significance tables for ontology-based function prediction.

# Dependencies
`text_representations/text_representations.yml` (environment `HOPER_textrepresentations`, created by `create_env.sh`).

## Data

The function prediction data should be provided in CSV format with the following columns:

- Model: The name of the prediction model.
- Measure: The evaluation measure used (e.g., F1-Weighted, Accuracy, Precision).
- Aspect: The functional aspect (BP, CC, or MF).
- Value: The value of the evaluation measure for each model and aspect.

Please make sure the data is formatted correctly before running the scripts.

### Options

The script supports the following command-line options:

- `-f` or `--figures`: Create figures from the results.
- `-s` or `--significance`: Create significance tables from the results.
- `-rfp` or `--resultfilespath`: Path for the result files (required).
- `-a` or `--all`: Create both figures and significance tables.

At least one option should be selected. If no options are provided, an error message will be displayed.

### How to Run

`bash download_data.sh` places the result files in `result_files/results/` (see the main [README](../../README.md#installation)).
Run from the repository root, either with the launcher (`choice_of_module: [text]`, `choice_of_process: [visualize]`)
or directly:

```shell
conda activate HOPER_textrepresentations
python text_representations/result_visualization/visualize_results.py -a -rfp ./text_representations/result_visualization/result_files/results/
```

Use `-f` for the figures only and `-s` for the significance tables only.

## Definition of Output

The script will load the results and perform the selected actions based on the provided options. The output will be generated in the following manner:

- If the `-f` or `--figures` option is selected, figures will be created and saved.
- If the `-s` or `--significance` option is selected, significance tables will be created and saved.
- If the `-a` or `--all` option is selected, both figures and significance tables will be created and saved.
- The `figures` directory will contain the generated figures in PNG format.
- The `significance` directory will contain the calculated significance scores in CSV format.

## License

Copyright (C)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
