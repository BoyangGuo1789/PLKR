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
    interested_folders = {"base2new"}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    experiment_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )

    for root, dirs, files in os.walk(input_folder):

        if "log.txt" in files and any(folder in root for folder in interested_folders):
            print(f"Processing file: {root}/log.txt")
            parts = root.split(os.sep)
            (
                experiment_name,
                experiment_mode,
                dataset_name,
                hyperparameter,
                trainer_name,
                config_name,
            ) = parts[-6:]

            log_path = os.path.join(root, "log.txt")
            extracted_lines = extract_info_from_log(log_path)

            experiment_data[experiment_name][experiment_mode][hyperparameter][
                trainer_name
            ][config_name].append((dataset_name, extracted_lines))

    for experiment_name, experiment_modes in experiment_data.items():
        for experiment_mode, hyperparameters in experiment_modes.items():
            for hyperparameter, trainer_names in hyperparameters.items():
                for trainer_name, config_names in trainer_names.items():
                    for config_name, dataset_infos in config_names.items():
                        output_dir = os.path.join(
                            output_folder,
                            experiment_name,
                            experiment_mode,
                            hyperparameter,
                            trainer_name,
                            config_name,
                        )
                        os.makedirs(output_dir, exist_ok=True)

                        all_dataset_log_path = os.path.join(
                            output_dir, "all_dataset_log.txt"
                        )
                        with open(all_dataset_log_path, "w", encoding="utf-8") as f:
                            for dataset_name, lines in dataset_infos:
                                f.write(f"=== Dataset: {dataset_name} ===\n")
                                f.write("\n".join(lines))
                                f.write(
                                    "\n--------------------------------------------------\n"
                                )

if __name__ == "__main__":
    main()
