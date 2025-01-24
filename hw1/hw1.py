# Andrew Kelton
# January 25, 2025
# EEL4872 Spring 2025
# Professor Gurupur
# Assignment 1

# Markov Chain Model

import random
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Bold letters
BOLD = '\033[1m'
END = '\033[0m'

current_date = datetime.now()

# Format date
PREVIOUS_DATE = (current_date - relativedelta(years=1)).strftime("%Y-%m-%d")
PREVIOUS_TOMORROW_DATE = (current_date - relativedelta(years=1, days=-1)).strftime("%Y-%m-%d")
CURRENT_DATE = formatted_date = current_date.strftime("%Y-%m-%d")
TOMORROW_DATE = (current_date + relativedelta(days=1)).strftime("%Y-%m-%d")

'''
Default forecasts, automatically all the same 
probability to be each state from each current state.
Last year's forecast simulated first to get last year's
probabilities for each state from each current state 
based on day's starting state. 
'''
INIT_PREVIOUS_YEAR_FORECASTS = {
    'sunny': [0.33, 0.33, 0.33],
    'cloudy': [0.33, 0.33, 0.33],
    'rainy': [0.33, 0.33, 0.33]
}

WEATHER_STATES = ['sunny', 'cloudy', 'rainy'] # Possible weather states

# Returns randomized next weather state from current state using weights in forecasts
def next_weather_state(current_state : str, forecasts : dict) -> str:
    return random.choices(WEATHER_STATES, forecasts[current_state])[0]

# Returns the forecast of the weather
def simulate_weather(init_weather : str, forecasts : dict, predictions=10) -> dict:
    weather = [init_weather]
    current_state = init_weather

    # Run prediction 'prediction(s)' times
    for _ in range(predictions - 1):
        current_state = next_weather_state(current_state, forecasts)
        weather.append(current_state)
    return weather

# Print forecast probabilities/predictions from previous year's simulation
def print_forecast_probabilities(forecast_probabilities : dict, forecast : dict, init_state : str) -> None:    
    print(BOLD + f'''{PREVIOUS_TOMORROW_DATE} Predicted Weather Probabilties when it was {init_state} on {PREVIOUS_DATE}''' + END)
    print(*(f"{key}: {value}" for key, value in forecast_probabilities.items()), sep="\n")
    print(f'''Predictions for {PREVIOUS_TOMORROW_DATE}:''', forecast)
    print(BOLD +f'''\nTesting Each State on {CURRENT_DATE} Against {PREVIOUS_DATE}'s State''' + END)

# Main
def main():

    # Run random process simulation for all 3 initial states
    # Initial states are yesterday's state from last year
    for init_state in WEATHER_STATES:
        print()

        forecast = simulate_weather(init_state, INIT_PREVIOUS_YEAR_FORECASTS) # Get previous year's forecast

        # Collect transition counts
        transition_counts = {state: {next_state: 0 for next_state in WEATHER_STATES} for state in WEATHER_STATES}

        # Count transitions
        for i in range(len(forecast) - 1):
            current_state = forecast[i]
            next_state = forecast[i + 1]
            transition_counts[current_state][next_state] += 1

        # Normalize to probabilities
        transition_probabilities = {}
        for state, transitions in transition_counts.items():
            total_transitions = sum(transitions.values())

            if total_transitions > 0:

                # Add to the new transition dictionary
                transition_probabilities[state] = {
                    next_state: count / total_transitions
                    for next_state, count in transitions.items()
                }
            else:
                
                # Default if no transitions
                transition_probabilities[state] = {
                    next_state: 1 / len(WEATHER_STATES) for next_state in WEATHER_STATES
                }

        # Create dictionary of last year's actual forecast
        simulated_forecast_probabilities = {
            state: [transition_probabilities[state][next_state] for next_state in ['sunny', 'cloudy', 'rainy']]
            for state in ['sunny', 'cloudy', 'rainy']
        }
        
        # Print probabilites and predictions determines from last year same day
        print_forecast_probabilities(simulated_forecast_probabilities, forecast, init_state)

        # Run the prediction for all 3 current states when it was 'init_state' last year 
        # today, and 'curr_state' this year today. Uses same forecast from initial 
        # simulation for all 3 current states.
        for curr_state in WEATHER_STATES:
            current_forecast = simulate_weather(curr_state, simulated_forecast_probabilities, 2)

            # Prediction for tomorrow's weather based on last year's predictions for same day last year
            print(f'{CURRENT_DATE} is {curr_state}, {TOMORROW_DATE} it will be {current_forecast[1]}')

        print(BOLD + '\n-------------------------------------------------------------------------------------' + END)


if __name__ == '__main__':
    main()
