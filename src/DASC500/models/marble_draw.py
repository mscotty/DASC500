import random
from collections import Counter

def marble_draw(marbe_colors=['R', 'G', 'B'], 
               num_simulations=100_000,
               output_file=None):
    """!
    @brief Simulate random marble draws from a bag where the order matters and the order does not

    @param[in] marble_colors (list): List of different marble colors as a string
    @param[in] num_simulations (int): Total number of simulations to run
    @param[in] output_file (str): File path to the output file
    """
    # Simulate draws for ordered pairs (where the order of the marble drawn is significant)
    ordered_results = [(random.choice(marbe_colors), random.choice(marbe_colors)) for _ in range(num_simulations)]
    ordered_counts = Counter(ordered_results)

    # Calculate probabilities for ordered pairs
    ordered_probabilities = {outcome: count / num_simulations for outcome, count in ordered_counts.items()}

    # Generate the string to present the ordered pairs result
    prt_str = ""
    prt_str += "Ordered Pair Probabilities:\n"
    for outcome, prob in sorted(ordered_probabilities.items()):
        prt_str += f"{outcome}: {prob:.4f}\n"

    # Simulate draws for unordered pairs
    unordered_results = [tuple(sorted((random.choice(marbe_colors), random.choice(marbe_colors)))) for _ in range(num_simulations)]
    unordered_counts = Counter(unordered_results)

    # Calculate probabilities for unordered pairs
    unordered_probabilities = {outcome: count / num_simulations for outcome, count in unordered_counts.items()}

    # Generate the string to present the unordered pairs result
    prt_str += "\nUnordered Pair Probabilities:\n"
    for outcome, prob in sorted(unordered_probabilities.items()):
        prt_str += f"{outcome}: {prob:.4f}\n"
    
    # Print to the output_file (if one is provided) or print out to a the command prompt
    if output_file is None:
        print(prt_str)
    else:
        with open(output_file, 'w') as f:
            f.write(prt_str)