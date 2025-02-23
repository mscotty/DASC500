# Compute Thickness-to-Chord Ratio
def thickness_to_chord_ratio(thickness, chord_length):
    """
    Compute the maximum thickness-to-chord ratio.
    
    @param thickness: Thickness distribution.
    @param chord_length: Chord length of the airfoil.
    @return: Maximum thickness-to-chord ratio.
    """
    return max(thickness) / chord_length