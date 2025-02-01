'''
This file generates a csv containing all the possible parameters to work with create_graphs
'''
import pandas as pd


'''Pulled in ffrom parameter_generator, in fact we should go back and change that code to pull these values in from here.'''
# Game-specific setup
total_occupancy_high_setting = [0.8, 0.9, 0.95, 0.98, 0.99, 1, 1.01, 1.02, 1.05, 1.1, 1.2]
game_modes = ["expected", "top_k"]

# Player-relevant variables
win_values_underdog = [i / 100 for i in range(5, 51, 5)]  # 0.05, 0.1, ..., 0.5
blind_combos_0 = [False, False]
blind_combos_1 = [False, True]
levels_0 = [100, 100]
levels_1 = [0, 100]

# Category-relevant variables
log_or_normal = ['log', 'normal']
percent_pop_high_mean = [i / 10 for i in range(1, 5)]  # 0.1, 0.2, ..., 0.4
high_low_ratio_means = [i / 10 for i in range(12, 21, 2)]  # 1.2, 1.4, ..., 2.0
high_low_ratio_variances = [i / 10 for i in range(12, 21, 2)]
mean_variance_ratios = [i / 100 for i in range(50, 151, 25)]  # 0.5, 0.75, ..., 1.5
percent_pop_high_variance = [i / 10 for i in range(1, 5)]

# Now we need to create a dataframe to store each value on a line with its affiliated dimension name

df=pd.DataFrame({
    'dimension': ["pct_total", "game_mode", "win_value_underdog", "blind_combo_0","blind_combo_1", "level_0", "level_1", "lognormal","pct_high_mean", "high_low_ratio_mean", "high_low_ratio_variance","mean_variance_ratio", "pct_high_sigma"],
    'values': [total_occupancy_high_setting, game_modes, win_values_underdog, blind_combos_0, blind_combos_1, levels_0, levels_1, log_or_normal, percent_pop_high_mean, high_low_ratio_means, high_low_ratio_variances, mean_variance_ratios, percent_pop_high_variance]
})

# Print the dataframe to verify the values and dimensions
print(df)


# If everything looks good, you can save the dataframe to a CSV file for further use or analysis

df.to_csv('dimension_values.csv', index=False)