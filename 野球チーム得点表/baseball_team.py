class Baseball_Team:
    def __init__(self,name,win,lose,draw):
        self.name = name
        self.win = win
        self.lose = lose
        self.draw = draw

    def calc_win_rate(self):
        return self.win / (self.win + self.lose)
    
    def show_team_result(self):
        print(f"{self.name:8s} {self.win:3d}{self.lose:5d} {self.draw:5d}    {self.calc_win_rate():.3f}")

