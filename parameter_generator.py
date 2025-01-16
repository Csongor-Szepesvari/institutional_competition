'''
The task of this file is to set up the conditions for each type of game we're investigating
'''
import os
import csv

# Game-specific setup
total_occupancy_high_setting = [0.8, 0.9, 0.95, 0.98, 0.99, 1, 1.01, 1.02, 1.05, 1.1, 1.2]
game_modes = ["expected", "top_k"]

# Player-relevant variables
win_values_underdog = [i / 100 for i in range(5, 51, 5)]  # 0.05, 0.1, ..., 0.5
blind_combos = [[False, False], [False, True]]
levels = [[100, 0], [100, 100]]

# Category-relevant variables
log_or_normal = ['log', 'normal']
percent_pop_high_mean = [i / 10 for i in range(1, 5)]  # 0.1, 0.2, ..., 0.4
high_low_ratio_means = [i / 10 for i in range(12, 21, 2)]  # 1.2, 1.4, ..., 2.0
high_low_ratio_variances = [i / 10 for i in range(12, 21, 2)]
mean_variance_ratios = [i / 100 for i in range(50, 151, 25)]  # 0.5, 0.75, ..., 1.5
percent_pop_high_variance = [i / 10 for i in range(1, 5)]

# Create a subfolder for storing CSV files
output_folder = "not_started"
os.makedirs(output_folder, exist_ok=True)

# Initialize variables
combinations = []
file_counter = 1

# Iterate through all parameter combinations
for pct_total in total_occupancy_high_setting:
    for game_mode in game_modes:
        for win_value_underdog in win_values_underdog:
            for blind_combo in blind_combos:
                for level in levels:
                    for lognormal in log_or_normal:
                        for pct_high_mean in percent_pop_high_mean:
                            for high_low_ratio_mean in high_low_ratio_means:
                                for high_low_ratio_variance in high_low_ratio_variances:
                                    for mean_variance_ratio in mean_variance_ratios:
                                        for pct_high_sigma in percent_pop_high_variance:
                                            # Collect combination
                                            combinations.append([
                                                pct_total, game_mode, win_value_underdog, blind_combo[0],
                                                blind_combo[1], level[0], level[1], lognormal,
                                                pct_high_mean, high_low_ratio_mean, high_low_ratio_variance,
                                                mean_variance_ratio, pct_high_sigma
                                            ])

                                            # Write to CSV every 100 combinations
                                            if len(combinations) == 100:
                                                filename = os.path.join(
                                                    output_folder,
                                                    f"params_file_{file_counter:03d}_occupancy{pct_total}_mode{game_mode}.csv"
                                                )
                                                with open(filename, mode='w', newline='') as file:
                                                    writer = csv.writer(file)
                                                    writer.writerow([
                                                        "pct_total", "game_mode", "win_value_underdog", "blind_combo_0",
                                                        "blind_combo_1", "level_0", "level_1", "lognormal",
                                                        "pct_high_mean", "high_low_ratio_mean", "high_low_ratio_variance",
                                                        "mean_variance_ratio", "pct_high_sigma"
                                                    ])
                                                    writer.writerows(combinations)

                                                # Clear combinations and increment file counter
                                                combinations = []
                                                file_counter += 1

# Write remaining combinations to a final CSV (if any)
if combinations:
    filename = os.path.join(
        output_folder,
        f"params_file_{file_counter:03d}_remaining.csv"
    )
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "pct_total", "game_mode", "win_value_underdog", "blind_combo_0",
            "blind_combo_1", "level_0", "level_1", "lognormal",
            "pct_high_mean", "high_low_ratio_mean", "high_low_ratio_variance",
            "mean_variance_ratio", "pct_high_sigma"
        ])
        writer.writerows(combinations)
