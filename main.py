'''
This file's job is to contain a single function that processes a single row from a dataframe that outlines one experiment

The tasks are as follows:
1. Generate all of the objects necessary from the row description
2. Find the equilibrium
3. Simulate game 1000 times
'''

from objects import *
#import time as t

def process_row(row, verbose=False):
    '''
    Process row takes a row from a dataframe, runs an experiment, simulates the outcome and returns the results
    '''
    if verbose:
        print(row)
        print()


    start = t.time() # just measuring the amount of time one run takes
    '''
    *** TASK 1 ***
    Process a row and generate objects

    Generator from objects, the functions:
        - generate_players(blind_combo, level, win_value_underdog)
        - categories_generator(high_low_ratio_mean, high_low_ratio_variance, mean_variance_ratio, high_mean_probability, high_variance_probability)
    '''
    players_list = generate_players(blind_combo=[row['blind_combo_0'], row['blind_combo_1']], level=[row['level_0'], row['level_1']], win_value_underdog=row['win_value_underdog'])
    categories_dict = categories_generator(high_low_ratio_mean=row['high_low_ratio_mean'], high_low_ratio_variance=row['high_low_ratio_variance'], mean_variance_ratio=row['mean_variance_ratio'], high_mean_probability=row['pct_high_mean'], high_variance_probability=row['pct_high_sigma'], log_normal=row['lognormal'])

    if verbose:
        print()
        print("population distribution, mean, std over categories:", [(category.get_size(), category.get_mean(), category.get_std()) for category in categories_dict.values()])

    size = sum([category.get_size() for category in categories_dict.values()]) # find the rounded number of students

    #print("total population:", size)
    to_admit = int((row['pct_high_mean']*row['pct_total']*size)//2) # find how many to admit, its relative to the high mean population

    #print("number each agent is seeking to admit:", to_admit)


    #print()

    if row['game_mode'] == 'expected': # if its expected them top_k is none
        top_k = None
    else:
        top_k = int(to_admit*0.2)
    game = Game(num_players=len(players_list), to_admit=to_admit, players=players_list, categories=categories_dict, game_mode_type=row['game_mode'], top_k=top_k, log_normal=row['lognormal'], verbose=verbose)
    '''
    *** TASK 2 FIND THE EQUILIBRIUM***
    '''
    game.find_strategies_iterated_br()

    if verbose:
        print(game.get_strat_list())

    players = {}
    for player in players_list:
        players[player.name] = player
    # player dictionary generation functions correctly
    #print("Checking players dictionary")
    #print(players)
    categories = game.categories
    to_admit = game.to_admit
    game_mode = game.game_mode_type
    top_k = game.top_k
    log_normal = game.log_normal


    '''
    *** TASK 3 SIMULATE AND RETURN RESULTS ***
    '''
    num_runs = 1000

    results = np.zeros(num_runs)
    for _ in range(num_runs):
        # simulate_game returns a dictionary that is key:value, player.name:{raw_util, percent_total_util}, we will return the average of percent_utils over 1000 runs
        outcome = simulate_game(players=players, categories=categories, to_admit=to_admit, game_mode=game_mode, top_k=top_k, log_normal=log_normal, verbose=verbose)
        #print(outcome)
        #print()
        results[_] = outcome['p1']['pct_total_util']
    
    mean = np.mean(results)
    std = np.std(results)
    end = t.time()
    
    if verbose:
        print()
        print("Evaluation:")
        print(f'{num_runs} runs resulted in a mean total utility achieved of {mean} and standard deviation of {std}.')
        print(f'Took {end-start} seconds to run one experiment with {num_runs} simulations at the end.')
    return mean, std

'''
import pandas as pd
import os

file_path = os.path.join(os.getcwd(), 'not_started\params_file_11218_occupancy0.98_modetop_k.csv')
df = pd.read_csv(file_path)

#print(df)

#for i in range(df)

#print(process_row(df.iloc[98], verbose=False))
#start = t.time()
#df[["underdog_mean", "underdog_variance"]] = df.apply(process_row, axis=1, result_type='expand')
#print(t.time()-start, "seconds to process a whole file")
#print(df)
'''