'''
This file contains all of our objects and functions we will use to go through to find equilibrium points for different games and then simulate their outcomes. 
'''
import numpy as np
import time as t
from scipy.stats import norm

def calculate_replicates_for_top_k(
    params,          # List of (dist_type, mu, sigma, sample_size) tuples
    k,               # Number of top-k values to consider
    dist_type,       # Whether we're calculating for normal or lognormal
    confidence=0.95,      # Desired confidence level (e.g., 0.95 for 95% confidence)
    error_rate=0.01,      # Allowed relative error (e.g., 0.01 for 1% error)
    max_iters=50,    # Max iterations for convergence
    tol=1e-3         # Tolerance for replicate change
):
    """
    Calculate the number of replicates required for estimating the mean of the top-k
    values from a fixed-size combined sample of distributions.

    Parameters:
        params (list of tuples): Each tuple (dist_type, mu, sigma, n) specifies:
                                 - dist_type: 'normal' or 'lognormal'
                                 - mu: mean of the distribution
                                 - sigma: standard deviation of the distribution
                                 - n: number of samples to draw from the distribution
        k (int): Number of top-k values to estimate.
        confidence (float): Desired confidence level (e.g., 0.95).
        error_rate (float): Allowed relative error in the top-k mean estimate.
        max_iters (int): Maximum number of iterations to converge.
        tol (float): Tolerance for replicate count change between iterations.

    Returns:
        int: Estimated required number of replicates.
    """
    # Calculate z-score for the confidence level
    z = norm.ppf(1 - (1 - confidence) / 2)
    
    # Initial guess for number of replicates
    num_replicates = 100
    previous_replicates = 0

    # Total combined sample size from all distributions
    combined_sample_size = sum(n for n, _, _ in params)
    k = int(min(combined_sample_size, k))

    for iteration in range(max_iters):
        # Generate the combined sample based on specified sample sizes and distribution types
        combined_data = []
        for n, mu, sigma in params:
            if dist_type == 'normal':
                combined_data.extend(np.random.normal(mu, sigma, int(n)))
            elif dist_type == 'log':
                combined_data.extend(np.random.lognormal(mu, sigma, int(n)))
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")
        
        combined_data = np.array(combined_data)
        
        # Extract top-k values
        top_k_values = np.sort(combined_data)[-k:]
        top_k_mean = np.mean(top_k_values)
        top_k_std = np.std(top_k_values, ddof=1) if k>1 else np.std(top_k_values)
        
        # Estimate the required number of replicates
        #print(z, top_k_std, error_rate, top_k_mean)
        required_replicates = max(1,int(((z * top_k_std) / (error_rate * top_k_mean))**2))
        #print(required_replicates)
        # Convergence check
        if abs(required_replicates - num_replicates) / num_replicates < tol:
            return required_replicates
        
        # Update replicate count for the next iteration
        previous_replicates = num_replicates
        num_replicates = required_replicates

    # If convergence is not achieved, return the last estimate
    return num_replicates


def transform_mu_sigma_to_log(mu, sigma):
    sigma_log = np.log(sigma)/2
    mu_log = np.log(mu)/2
    return mu_log, sigma_log

# COMPLETED
def generate_samples_top_k(categories, log_or_normal, top_k=None):
    '''
    categories : tuple of tuples, ((number samples, mu, sigma), (number_samples, mu, sigma), ...)
    categories : has shape (num categories, 3)
    '''
    outcomes = np.array([])
    for category in categories:
        rng = np.random.default_rng()
        if log_or_normal == 'log':
            outcomes = np.concatenate((outcomes, rng.lognormal(mean=category[1], sigma=category[2], size=int(category[0]))))
        elif log_or_normal == 'normal':
            outcomes = np.concatenate((outcomes, rng.normal(loc=category[1], scale=category[2], size=int(category[0]))))

    if top_k is not None:
        outcomes = np.sort(outcomes)[::-1][:top_k]
    return outcomes

# Now we will create a function that takes a memo and returns the expected value for a certain distribution.

def eval_particular_distribution(categories, log_or_normal, memo, top_k=None, verbose=False, confidence=0.95, error=0.05):
    if verbose:
        print("testing", (categories, log_or_normal, top_k))
        print()

    # CONVERT TOP_K TO BE AT MOST AS LARGE AS THE TOTAL SIZE OF SAMPLES
    total_samples = sum([samples for samples, mu, sigma in categories])
    if top_k:
        top_k = int(min(top_k, total_samples))

    if (categories, log_or_normal, top_k) in memo:
        if verbose:
            print("Found in the memo already, time saved yay!")
            print()
        return memo[(categories, log_or_normal, top_k)]

    # Calculate the number of samples required to get within our confidence and error regions
    num_runs_estimate = calculate_replicates_for_top_k(params=categories, k=top_k, dist_type=log_or_normal, error_rate=0.05)
    if verbose:   
        print("Estimate for the number of runs needed:", num_runs_estimate)
    if log_or_normal=='log':
        num_runs = max(1000,min(50000, num_runs_estimate))
    else:
        num_runs = max(100,min(5000, num_runs_estimate))


    total_outcome = np.array([])
    start = t.time()
    for i in range(num_runs):
        total_outcome = np.concatenate((total_outcome, np.array([np.mean(generate_samples_top_k(categories, log_or_normal, top_k=top_k))])))
        if verbose:
            if ((i+1)/num_runs*100/20).is_integer():
                print(f"{(i+1)/num_runs*100}% completed sampling one iteration.")
    #if verbose:
        #print(total_outcome)
    memo[(categories, log_or_normal, top_k)] = total_outcome.mean()
    if verbose:
        print(len(memo))
        print(t.time()-start, "seconds to find the result of a 1 incrementation.")
        print()
    return memo[(categories, log_or_normal, top_k)]

def convert_category_strategy_to_evaluator(categories, strategy):
    # takes a categories and strategy as used normally and converts it into tuples to be used with the sampler
    n = len(categories)
    new_categories = np.zeros((n,3))
    #print(new_categories)
    # categories in this case refers a dictionary of {"cat_name":category}
    for i, category in zip(range(n),categories.values()):
        new_categories[i][0] = int(strategy[category.get_name()])
        new_categories[i][1] = category.get_mean()
        new_categories[i][2] = category.get_std()

    new_categories = tuple(tuple(row) for row in new_categories)
    return new_categories


class Category():
    def __init__(self, name, mean, std, size, log_or_normal):
        
        self.mean = mean
        self.std = std
        if log_or_normal == 'log':
            self.mean, self.std = transform_mu_sigma_to_log(self.mean, self.std)
        self.size = size
        self.name = name
        self.log_normal = log_or_normal

    def get_samples(self, n:int):
        if n > self.size:
            raise ValueError("Error: we can't sample more than there are elements in this category.")
        else:
            rng = np.random.default_rng()
            if self.log_normal == 'log':
                return rng.lognormal(mean=self.mean, sigma=self.std, size=n)
            return rng.normal(loc=self.mean, scale=self.std, size=n)

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def get_size(self):
        return self.size

    def get_name(self):
        return self.name

class Player():
    def __init__(self, win_value:float, blind:bool, level:int, name:str):
        '''
        Initializer for a Player object, takes in the following parameters:
        win_value: the value used to calculate the probability of victory for player collisions
        blind: a boolean value that determines if the player is blind or not
        level: an integer value that determines the level of the player, with 0 not optimizing past the single-player level and 1+ optimizing for multiple iterations
        '''
        self.strategy = {"Q1":0, "Q2":0, "Q3":0, "Q4":0}
        self.blind_strategy = {"high":0, "low":0}
        self.win_value = win_value
        self.blind = blind
        self.level = level
        self.name = name
        

    def update_blind_strategy(self, strategy, game):
        '''
        To update the blind strategy with these categories we have to multiply the probabilities with the category sizes to get the total occupancy and convert it into low and high
        '''
        #print(game.categories)
        #print(strategy)
        q3 = game.categories["Q3"] # was put in for testing purposes
        self.blind_strategy["high"] = strategy["Q1"] * game.categories["Q1"].get_size() + strategy["Q2"] * game.categories["Q2"].get_size() / (game.categories["Q1"].get_size() + game.categories["Q2"].get_size())
        self.blind_strategy["low"] = strategy["Q3"] * q3.get_size() + strategy["Q4"] * game.categories["Q4"].get_size() / (q3.get_size() + game.categories["Q4"].get_size())


    def update_strategy(self, blind_strategy, game):
        '''This method updates the player strategy based on the blind strategy given'''

        # we need to ensure that the strategies we're transferring are translating to integers

        # we need to find the fixed number then distribute relative to their ratios
        '''
        to_admit_high = blind_strategy['high']*(game.categories["Q1"].get_size() + game.categories["Q2"].get_size())
        assert to_admit_high.is_integer(), "to_admit_high is not an integer, instead it is" + str(to_admit_high)

        to_admit_low = blind_strategy['low']*(game.categoriess["Q3"].get_size() + game.categories["Q4"].get_size())
        assert to_admit_low.is_integer(), "to_admit_low is not an integer, instead it is" + str(to_admit_low)
        '''
        # if we've passed the asserts then we can figure out how to distribute amongst the strategies if we can't get a perfect split based on the ratio between high and low
        # we multiply the category size by the blind strategy and we round off to the nearest integer, then divide down again
        self.strategy["Q1"] = np.round(blind_strategy["high"]*game.categories["Q1"].get_size(), 0)/game.categories["Q1"].get_size()
        self.strategy["Q2"] = np.round(blind_strategy["high"]*game.categories["Q2"].get_size(), 0)/game.categories["Q2"].get_size()
        self.strategy["Q3"] = np.round(blind_strategy["low"]*game.categories["Q3"].get_size(), 0)/game.categories["Q3"].get_size()
        self.strategy["Q4"] = np.round(blind_strategy["low"]*game.categories["Q4"].get_size(), 0)/game.categories["Q4"].get_size()


    def calculate_win_chance(self, players):
        win_percent = self.win_value / (sum([player.get_win_value() for player in players]))
        return win_percent

    def calc_expected_attendees(self, strategy, game):
        '''
        This method calculates the expected attendees from a particular strategy in context of all other player's strategies
        '''

        other_players = [player for player in game.players if player != self]  # gets a list of other players from a game
        #print(other_players)
        achieved_result_pct = {}
        for category in game.categories.values():
            # this for loop grabs a category and then calculates what percentage of the category we will successfully absorb
            # we do this by subtracting the percentage we lose to other players
            achieved_result_pct[category.get_name()] = strategy[category.get_name()] * (1 - self.calculate_pct_lost_to_others(other_players, category.get_name()))
        return achieved_result_pct


    def calculate_pct_lost_to_others(self, other_players, category):
        # THIS METHOD NEEDS TO BE REWORKED
        '''
        This method returns the percentage of a category lost to others regardless of our own strategic occupation.
        Parameters:
        1. other_players: a list of all other players in the game
        2. category: the category we're looking at
        '''
        def helper(occupancy_list, index):

            _occ = occupancy_list
            #print(_occ)
            # start adding up the probabilities of the event
            prob_event = 1
            # include things in the denominator
            denom = self.win_value

            # check if we're at the bottom
            if index == len(_occ)-1:
                # calculate the value of this event
                for i in range(len(_occ)):
                    for j in range(len(other_players)):
                        # grab if we are occupying this element in the list
                        occupies = _occ[i]
                        if occupies:
                            prob_event *= other_players[j].strategy[category]
                            denom += other_players[j].win_value
                        else:
                            prob_event *= (1-other_players[j].strategy[category])
                # actual value calculation for a loss
                return prob_event * (1 - self.win_value/denom)

            else:
                _index = index + 1
                false_val = helper(occupancy_list=_occ, index=_index)
                _occ[_index] = True
                true_val = helper(occupancy_list=_occ, index=_index)
                return false_val + true_val

        f_start = [False for i in range(len(other_players))]
        #print(f_start)
        f_val = helper(f_start, 0)
        #print(f_val)
        f_start[0] = True
        t_val = helper(f_start, 0)

        return f_val + t_val

    def greedy_top_k_br(self, game, feasible_strategy_numbers:dict[str:float], k:int, blind=False):
        '''
        for each possible category we create a new strategy of incrementing each one by one
        then we check what the outcome of this combination is
        then we pick the maximal one (greedy)
        '''
        to_admit = game.to_admit
        # strategy of actual admittees
        new_strategy = {category.get_name():0 for category in game.categories.values()}
        if not blind:
            for i in range(to_admit):
                max_val = float('-inf')
                max_cat = ""
                for category in game.categories.values():
                    # we do the actual picking here
                    if 1 + new_strategy[category.name] <= feasible_strategy_numbers[category.name]:
                        name = category.get_name()
                        temp_strat = new_strategy.copy()
                        temp_strat[name] += 1
                        # this is where the difference needs to be made
                        category_tuple = convert_category_strategy_to_evaluator(categories=game.categories, strategy=temp_strat)
                        _val_strat = eval_particular_distribution(categories=category_tuple, top_k=k, log_or_normal=game.log_normal, memo=game.memo, verbose=game.verbose)
                        if _val_strat > max_val:
                            max_val = _val_strat
                            max_cat = name

                new_strategy[max_cat] += 1

            self.strategy = {category.get_name():new_strategy[category.get_name()]/feasible_strategy_numbers[category.get_name()] for category in game.categories.values()}

            #print("line 171 update", self.strategy)
            self.update_blind_strategy(strategy=self.strategy, game=game)

        else:
            # the blind agent just takes the high mean players
            high_nums = feasible_strategy_numbers["Q1"] + feasible_strategy_numbers["Q2"]
            low_nums = feasible_strategy_numbers["Q3"] + feasible_strategy_numbers["Q4"]

            high_admit = min(to_admit, high_nums)
            remainder = to_admit-high_admit

            blind_strategy = {"high":high_admit/high_nums, "low":remainder/low_nums}
            self.update_strategy(blind_strategy=blind_strategy, game=game)





    def best_response(self, game):
        assert type(game) == Game
        '''
        This method calculates the best response in response to a game class and updates the strategy based on the best response type
        '''
        max_admit = game.to_admit
        admits = 0

        # calculate the maximum number of students we can get
        max_strat = {}
        for category in game.categories.values():
            max_strat[category.get_name()] = 1
        feasible_strat = self.calc_expected_attendees(strategy=max_strat, game=game)


        feasible_strategy_numbers = {}
        for category in game.categories.values():
            feasible_strategy_numbers[category.get_name()] = feasible_strat[category.get_name()]*category.get_size()


        if game.game_mode_type == "expected":
            # use the expected case algorithm to evaluate what the expected best response would be in this setting

            # what are we trying to maximize in this setting?
            # c(game) = sum(attendees) - (num_attendees-desired_attendees)^2
            # first term wants you to maximize the number of students you admit, the second term caps it

            # in order to maximize this, we look at other players' allocation or occupancy in each category
            # and calculate how many admissions to give out in order of best to worst


            if not self.blind:
                new_strat = {category.get_name():0 for category in game.categories.values()}
                # follow the simple logic of increasing admittances to Q1 > Q2 > Q3 > Q4 etc

                    # !!! this logic is broken in case we're looking at log_normal distributions since the expectation is different
                    # We need to scale it so that the mean we supply results in the *desired mean*
                    # Lognormal distributions have an expectation = e ** (mu + sigma^2 / 2)
                for cat_name in ["Q1", "Q2", "Q3", "Q4"]:
                    if feasible_strategy_numbers[cat_name] <= max_admit:
                        # maxing out the category, so we just admit everyone
                        max_admit -= feasible_strategy_numbers[cat_name]
                        new_strat[cat_name] = 1
                    else:
                        # we can't max out the category, so we admit as many as desired and then stop
                        unrounded_strat = max_admit/feasible_strategy_numbers[cat_name]
                        # we need to modify this here to make it an integer (so long as its feasible)
                        new_strat[cat_name] = np.ceil(unrounded_strat * game.categories[cat_name].get_size())/game.categories[cat_name].get_size()
                        break

                # convert new strat numbers to percentages and then project

                self.strategy = new_strat
                self.update_blind_strategy(strategy=self.strategy, game=game)


            elif self.blind:
                # if our player is blind then we just have Q1 + Q2 > Q3 + Q4

                # so we add together category Q1 and Q2 as well as Q3 and Q4
                high_numbers = feasible_strategy_numbers["Q1"] + feasible_strategy_numbers["Q2"]
                low_numbers = feasible_strategy_numbers["Q3"] + feasible_strategy_numbers["Q4"]
                new_strat = {}

                # we want to admit in high, h
                high_admit = min(high_numbers, max_admit)
                low_admit = min(low_numbers, max_admit-high_admit)

                real_strat_high = high_admit/high_numbers
                real_strat_low = low_admit/low_numbers

                self.blind_strategy['high'] = real_strat_high
                self.blind_strategy['low'] = real_strat_low
                self.update_strategy(blind_strategy=self.blind_strategy, game=game)


        elif game.game_mode_type == "top_k":
            # use the top k algorithm to evaluate what the expected best response would be
            # c(game) = sum(top_k) - (num_attendees-desired_attendees)^2
            # first term wants you to maximize the number of students you admit, the second term caps it

            # in order to maximize this we keep building up students by student until the marginal value gained from having additional students is less than the chance of finding someone better
            # this is a greedy method

            # the perfect method is to literally find
            '''
            greedy top_k best response process
            '''
            self.greedy_top_k_br(game=game, feasible_strategy_numbers=feasible_strategy_numbers, k=game.top_k, blind=self.blind)

class Candidate():
    '''
    Candidate objects, has a value generated from a category.
    '''

    def __init__(self, category:Category):
        self.value = category.get_samples(1)
        self.competitors = []


    def add_competitor(self, player:Player):
        self.competitors.append(player)

    def simulate_winner(self)->Player:
        # randomly sample between 0-1

        # normalize their values and turn them into cumulative values
        comp_values = np.array([player.win_value for player in self.competitors])
        normalized_values = comp_values/np.sum(comp_values)
        #print(normalized_values)
        # set cumulative limits
        cumulative_probs = np.cumsum(normalized_values)
        #print(cumulative_probs)
        # get random value
        random_value = np.random.rand()
        # find winner
        for i in range(len(cumulative_probs)):
            if random_value < cumulative_probs[i]:
                return self.competitors[i]

class Game():
    '''
    Game class meant for creating specific instances of games, both to find equilibrium points and also to simulate those games
    '''
    def __init__(self, num_players:int, to_admit: int, players:list[Player], categories:dict[str:Category], game_mode_type:str, top_k=None, log_normal=False, verbose=False):
        '''
        Initializes a game object based on:

            num_players->int : the number of players in a game
            to_admit->int : the number of students each player is admitting
            win_vals->list[float] : the associated "win value" with each player
            categories->list[Category] : a list of the categories of the students
            game_mode_type->str : the type of the game, either "top_k" or "expected"
        '''

        self.num_players = num_players
        self.players = players
        self.player_dict = {player.name:player for player in self.players}
        self.to_admit = to_admit
        self.categories = categories
        self.category_keys = ["Q1", "Q2", "Q3", "Q4"]
        #self.blind_categories = self.generate_blind_cat() DEPRECATED
        self.game_mode_type = game_mode_type
        self.top_k = top_k
        self.log_normal = log_normal
        self.verbose = verbose
        self.memo = {}

    def find_strategies_iterated_br(self):
        '''
        This method finds the stable strategies for each player based on iterated best response. If a stable profile is found it is a Nash Equilibrium

        The method works by iterating through the list of players and asking each off them to best respond to the game object as it stands.
        If after one iteration of going through all players, no players' strategies change, we have found a stable point and exit. The loop.

        Has no inputs. But the resultant strategies are updated in the strategy dictionaries of the players contained in the list.
        '''
        last_strats = None

        # while last strats aren't the same as the current updated list, loop (detects if there's no change from looping)

        # NOTE: straight equality (==) CAN be used here, because python does an element-wise equality check of lists, which then does an equality check on the nested dictionaries, very cool!
        looped = 0
        while last_strats != self.get_strat_list():
            # update last strats to the current strats before updating our strategies in the inner loop
            last_strats = self.get_strat_list()
            for player in self.players:
                if player.level >= looped:
                    player.best_response(self)
                #print(player.name, player.strategy)
            looped += 1
            #print()
        #print("Woohoo! Converged!")
        #print()

    def get_strat_list(self):
        '''
        This method loops through the player list and gets a list of their respective strategies
        '''
        return [player.strategy for player in self.players]
                
def simulate_game(players:dict[str:Player], categories:dict[str:Category], to_admit:int, game_mode:str, top_k:int, log_normal:str, verbose=False):
    '''
    This method simulates a game as it would actually play out based on the players' strategies.

    It steps through each category, generates random sets of admittees for each player based on their strategies, resolves collisions and gets actual attendants
    '''
    

    attendees = {player.name:[] for player in players.values()}
    for category in categories.values():
        #print("category size", category.get_size())
        # for each category generate a list of admittees
        candidates = [Candidate(category) for i in range(category.get_size())]
        # confirm that candidates is of the correct size
        #print("List of candidates", [candidate.value for candidate in candidates], "in", category.get_name())
        #print("Number of candidates", len(candidates), "category size (should be equal)", category.get_size())


        admittees = {player.name:set() for player in players.values()}

        # SIMULATE TRUE PLAYER SELECTION
        for player in players.values():
            # we want to highlight a selection of candidates, for each player we want to give 0 or 1 for each candidate
            gen = np.random.Generator(np.random.PCG64())
            # grab the number of Candidates the Player will try to secure in this category
            strat = int(np.round(player.strategy[category.get_name()]*category.get_size(), decimals=0))
            
            if verbose:
                print("strategy in", category.get_name(), "is to pull in", strat, "candidates")

            # confirm that the number of players to pull in is an integer
            if type(strat)==float:
                assert strat.is_integer(), "Strategy isn't integer based, need to trace where rounding didn't happen correctly"
            strat = int(strat)

            # this generates a set of numbers between 1 and category_size, with as many selections as was specified by our strategy, with no replacement
            # we set this to the admittee list
            admittees[player.name] = set(gen.choice(a=category.get_size(), size=strat, replace=False))

            # print out the selection
            if verbose:
                print("Player", player.name, "selection in category", category.name, "is:", admittees[player.name])



        '''
        Loop through candidates,
            then for each player
                Check if the given index is in admittees set for that player
                    if in: add player to candidate sweepstakes
            simulate the winner with candidate
            attendees[winning_name].append(candidate.value)
        Then simulate the chance breakdown and add the candidate value to the winning players attendees
        '''
        #print(candidates)
        for i in range(len(candidates)):
            candidate = candidates[i]
            for player in players.values():
                if i in admittees[player.name]:
                    candidate.add_competitor(player)
            if candidate.competitors != []:
                winning_player = candidate.simulate_winner()
                if verbose:
                    print(f"Candidate value is {candidate.value}")
                attendees[winning_player.name].append(candidate.value)
                if verbose:
                    print(f"Added {i}th attendee to", winning_player.name)
                    print(winning_player.name, "number of attendees is", len(attendees[winning_player.name]))
        if verbose:
            print()

    '''Calculate the utilities for the game and return it'''
    return get_game_utility(players=players, attendees=attendees, to_admit=to_admit, game_type=game_mode, top_k=top_k)

def get_game_utility(players:dict[str:Player], attendees:dict[str:float], to_admit:int, game_type:str, top_k=None)->dict[str:float]:
    '''evaluates a game based on our two utility functions and returns the utility of each player in a dictionary
    structure of dictionary is player.name : utility
    '''
    results = {player.name:{} for player in players.values()}
    if game_type=="top_k":
        # the utility function in this case will add up the top k values and subtract the difference with desired
        for player in players.values():
            utility = np.sum(sorted(attendees[player.name], reverse=True)[:top_k]) - (len(attendees[player.name])-to_admit)**2
            results[player.name]["raw_util"] = float(utility)
            results[player.name]["admitted"] = len(attendees[player.name])
    else:
        # straight sum minus the difference
        for player in players.values():
            #print(attendees[player.name], len(attendees[player.name]), to_admit)
            utility = np.sum(attendees[player.name]) - (len(attendees[player.name])-to_admit)**2
            #print(player.name, "had the goal of admitting", to_admit, "students, and actually admitted", len(attendees[player.name]), "students")
            results[player.name]["raw_util"] = float(utility)
            results[player.name]["admitted"] = len(attendees[player.name])

    total = sum([results[player]["raw_util"] for player in players.keys()])
    #print(total)
    for player_name in results.keys():
        results[player_name]["pct_total_util"] = results[player_name]['raw_util'] / total

    return results






def categories_generator(high_low_ratio_mean, high_low_ratio_variance, mean_variance_ratio, high_mean_probability, high_variance_probability, log_normal):
    '''
    This function generates categories based on the parameters provided. The parameters are as follows:
    high_low_ratio_mean: The ratio of the means of *conventionally* high talented students and conventionally low talented students.
    high_low_ratio_variance: The ratio of the variances of *unconventionally* high talented students and conventionally low talented students.
    mean_variance_ratio: The ratio of the mean to the variance of the students in each category. (Ex. A mean of 10 and a variance of 1 would have a mean_variance_ratio of 10).
    high_mean_probability: The probability that a student is *conventionally* high talented.
    high_variance_probability: The probability that a student is *unconventionally* high talented.
    '''
    categories = {}
    default_mean = 5
    epsilon = 1/100
    default_population = 120
    categories["Q1"] = Category(name="Q1", mean=default_mean*high_low_ratio_mean+epsilon, std=default_mean*mean_variance_ratio*high_low_ratio_variance, size=int(default_population*high_mean_probability*high_variance_probability), log_or_normal=log_normal)
    categories["Q2"] = Category(name="Q2", mean=default_mean*high_low_ratio_mean, std=default_mean*mean_variance_ratio, size=int(default_population*high_mean_probability*(1-high_variance_probability)), log_or_normal=log_normal)
    categories["Q3"] = Category(name="Q3", mean=default_mean+epsilon, std=default_mean*mean_variance_ratio*high_low_ratio_variance, size=int(default_population*(1-high_mean_probability)*high_variance_probability), log_or_normal=log_normal)
    categories["Q4"] = Category(name="Q4", mean=default_mean, std=default_mean*mean_variance_ratio, size=int(default_population*(1-high_mean_probability)*(1-high_variance_probability)), log_or_normal=log_normal)

    return categories


# temporary placeholder, will extend to more players later
def generate_players(blind_combo, level, win_value_underdog):
    players = []
    players.append(Player(blind=blind_combo[0], level=level[0], win_value=win_value_underdog, name="p1"))
    players.append(Player(blind=blind_combo[1], level=level[1], win_value=1-win_value_underdog, name="p2"))
    return players