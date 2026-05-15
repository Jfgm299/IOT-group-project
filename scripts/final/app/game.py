import random

rules = {"scissors": ("paper", "lizard"),
         "paper": ("rock", "rocknroll"),
         "rock": ("lizard", "scissors"),
         "lizard": ("rocknroll", "paper"),
         "rocknroll":("scissors", "rock")}

posibilities = list(rules.keys())

def input_validation(input):
    if input not in posibilities:
        return False
    
    return True

def match(player_choice, cpu_level: int = 1):
    
    if cpu_level == 0: # Always win
        cpu_choice = random.choice(rules[player_choice])
    elif cpu_level == 1: # Random
        cpu_choice = random.choice(posibilities)
    elif cpu_level == 999: # Always loose
        cpu_posibilites = []
        for k,v in rules.items():
            if player_choice in v:
                cpu_posibilites.append(k)
        
        cpu_choice = random.choice(cpu_posibilites)
    else: # Random
        cpu_choice = random.choice(posibilities)

    result = "-"
    if cpu_choice in rules[player_choice]:
        result = "Player"
    elif player_choice in rules[cpu_choice]:
        result = "AI"
    else:
        result = "Draw"

    return (result, cpu_choice)

if __name__ == "__main__":
    while True:
        player_choice = input(
        "Player, choose between: rock, paper, scissors, lizard and rocknroll (not spock), to defeat the evil AI: ")

        while not input_validation(player_choice):
            player_choice = input("Try again! (rock, paper, scissors, lizard and rocknroll): ")

        streak = 0

        result = match(player_choice, 999)

        while result == "Player" or result == "Draw":
            if result == "Player":
                streak += 1
                print(f"You won! One extra point. Total: {streak}")
            else:
                print(f"Draw! Try again. Current streak: {streak}")
            
            player_choice = input(
            "Player, choose between: rock, paper, scissors, lizard and rocknroll (not spock), to defeat the evil AI: ")

            while not input_validation(player_choice):
                player_choice = input("Try again! (rock, paper, scissors, lizard and rocknroll): ")
            
            match(player_choice)
    

        print(f"You loose... Max streak: {streak}")