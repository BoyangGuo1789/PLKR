import os
from collections import defaultdict

def extract_info_from_log(log_path):
    lines_to_save = {"accuracy": None, "max_accuracy": None}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "* accuracy:" in line:
                lines_to_save["accuracy"] = line.strip()
            elif "* Maximum value in accuracy:" in line:
                lines_to_save["max_accuracy"] = line.strip()

    if None in lines_to_save.values():
        return ["error"]
    return [lines_to_save["accuracy"], lines_to_save["max_accuracy"]]

def main():
    input_folder = "output_log"
    output_folder = "output_log_alldataset"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    experiment_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    desired_folders = ["evaluation", "imagenet"]

    for root, dirs, files in os.walk(input_folder):
        if (
            root.count(os.sep) == input_folder.count(os.sep) + 1
            and os.path.basename(root) in desired_folders
        ):
            for dir in dirs:
                current_path = os.path.join(root, dir)
                for subroot, _, subfiles in os.walk(current_path):
                    if "log.txt" in subfiles:
                        parts = subroot.split(os.sep)
                        (
                            experiment_mode,
                            trainer_name,
                            hyperparameter,
                            dataset_name,
                        ) = parts[-4:]

                        log_path = os.path.join(subroot, "log.txt")
                        extracted_lines = extract_info_from_log(log_path)

                        experiment_data[experiment_mode][trainer_name][
                            hyperparameter
                        ].append((dataset_name, extracted_lines))

    for experiment_mode, trainer_names in experiment_data.items():
        for trainer_name, hyperparameters in trainer_names.items():
            for hyperparameter, dataset_infos in hyperparameters.items():
                output_dir = os.path.join(
                    output_folder, experiment_mode, trainer_name, hyperparameter
                )
                os.makedirs(output_dir, exist_ok=True)

                all_dataset_log_path = os.path.join(output_dir, "all_datasets_log.txt")
                with open(all_dataset_log_path, "w", encoding="utf-8") as f:
                    for dataset_name, lines in dataset_infos:
                        f.write(f"=== Dataset: {dataset_name} ===\n")
                        f.write("\n".join(lines))
                        f.write(
                            "\n--------------------------------------------------\n"
                        )

if __name__ == "__main__":
    main()
