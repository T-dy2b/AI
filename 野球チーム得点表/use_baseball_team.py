from baseball_team import Baseball_Team

giants=Baseball_Team('Giants',77,64,2)
baystars=Baseball_Team('BayStars',71,69,3)
tigers=Baseball_Team('Tigers',69,68,6)
carp=Baseball_Team('Carp',70,70,3)
dragons=Baseball_Team('Dragons',68,73,2)
swallows=Baseball_Team('Swallows',59,82,2)

teams = [giants, baystars, tigers,carp, dragons, swallows]
print("team      win lose  draw   rete")
for team in teams:
    team.show_team_result()
    
