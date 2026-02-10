import random

def player(prev_play, opponent_history=[]):
    if prev_play != "":
        opponent_history.append(prev_play)

    # First move
    if len(opponent_history) == 0:
        return "R"

    # Helper maps
    counter = {"R": "P", "P": "S", "S": "R"}
    beats = {"R": "S", "P": "R", "S": "P"}

    # =========================
    # STRATEGY 1: Counter KRIS
    # =========================
    # Kris always counters our last move
    if len(opponent_history) >= 2:
        predicted = counter[opponent_history[-1]]
        return counter[predicted]

    # =========================
    # STRATEGY 2: Detect QUINCY pattern
    # =========================
    if len(opponent_history) >= 5:
        pattern = opponent_history[-5:]
        if pattern == ["R", "P", "S", "R", "P"]:
            return "S"
        if pattern == ["P", "S", "R", "P", "S"]:
            return "R"
        if pattern == ["S", "R", "P", "S", "R"]:
            return "P"

    # =========================
    # STRATEGY 3: Frequency counter (Abbey / Mrugesh)
    # =========================
    freq = {"R": 0, "P": 0, "S": 0}
    for move in opponent_history[-10:]:
        freq[move] += 1

    most_common = max(freq, key=freq.get)
    return counter[most_common]
