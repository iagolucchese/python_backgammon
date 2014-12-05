"""
	Neural network code, this is OUR OWN custom code
"""
from model import Board, WHITE, BLACK
import bpnn
import strategy
from game import Game, ComputerPlayer, ConsolePlayer

def otherColor(color):
	if (color == WHITE):
		return BLACK
	else:
		return WHITE

def evaluateMove(game,color):
	"""
	Produces the input array, and runs the NN function itself
	"""
	inputs = produce_NN_inputs(game,color)
	
	
	
def produce_NN_inputs(game, color):
	inputs = []
	#--- start of inputs ---
	for i in range(1,25): #for each position/triangle
		p = game.board.points[i].pieces
		for j in p: #for each checker in that position
			if (j.color == color) #if this is your color, fill the first 4 inputs, and add 4 empty ones at the end
				if (j.num >= 3):
					input.add((len(p)-3)/2)
					break
				else:
					inputs.add(1)
				inputs.add(0)
				inputs.add(0)
				inputs.add(0)
				inputs.add(0)
			else: #if this is NOT your color, add 4 empty inputs and then 4 to represent the checkers
				inputs.add(0)
				inputs.add(0)
				inputs.add(0)
				inputs.add(0)
				if (j.num >= 3):
					input.add((len(p)-3)/2)
					break
				else:
					inputs.add(1)
		#end for
	#end for
	if (color == WHITE): #add a input to determine if your color is white
		input.add(1)
	else:
		input.add(0)
	if (color == BLACK):#this input determines if you're playing with black
		input.add(1)
	else:
		input.add(0)
		
	enemyColor = otherColor(color)
	input.add(len(game.board.jailed(color))/2) #one half the amount of jailed pieces you have
	input.add(len(game.board.jailed(enemyColor))/2) #one half the amount of jailed pieces for the opponent
	input.add(len(game.board.homed(color)))
	input.add(len(game.board.homed(enemyColor)))
	#--- end of inputs ---
	return inputs

class NNPlayer(Player):
	"""
	Our own neural network player
	"""

	def __init__(I, color, nn):
		I.color = color
		I.neuralNetwork = nn
	def interact(I, game):
		high_score = -9999
		best_moves = []
		for moves in game.all_choices():
			copyGame = copy.deepcopy(game)
			copyGame.board = copyGame.board.copy()
			for m in moves:
				copyGame.move(m*)
			score = evaluateMove(game, I.color, I.neuralNetwork)
			#print("SCORE: {:5}	 PATH: {}".format(score, moves))
			if score > high_score:
				high_score = score
				best_moves = moves
		for move in best_moves:
			print("MOVE:", move)
			game.draw()
			game.move(*move)
			trainNetwork(high_score, best_moves,color,I.neuralNetwork)
	
if __name__ == '__main__':
    neuralNetwork = NN(198,50,1)
	
	game = Game()
	game.white = NNPlayer(WHITE, neuralNetwork)
	game.black = ComputerPlayer(BLACK, strategy.safe)