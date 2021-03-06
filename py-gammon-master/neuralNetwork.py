"""
	Neural network code, this is OUR OWN custom code
"""
from model import Board, WHITE, BLACK
import copy

import bpnn
import strategy
from game import Game, ComputerPlayer, ConsolePlayer

neuralNetwork = None

def otherColor(color):
	if (color == WHITE):
		return BLACK
	else:
		return WHITE

def evaluateMove(game,color):
	"""
	Gets the input array for the specific move, and runs the neural network with it
	"""
	inputs = produce_NN_inputs(game,color)
	result = neuralNetwork.update(inputs)
	return result

def trainNetwork(old_game,desired_output,color):
	"""
		Trains the neural network through temporal difference backpropagation
	"""
	old_inputs = produce_NN_inputs(old_game,color)
	patterns = [[old_inputs,[desired_output]]]
	#print patterns
	neuralNetwork.train(patterns) #accepts, in order, iterations, learning factor, and momentum as well
	
def produce_NN_inputs(game, color):
	"""
		This function should return an array with about 198 inputs, 
		mostly composed of 0's and 1's
	"""
	inputs = [0]*198
	#--- start of inputs ---
	for i in range(1,25): #for each position/triangle
		p = game.board.points[i].pieces
		if (game.board.points[i].color == WHITE): #if this is your color, fill the first 4 inputs, and append 4 empty ones at the end
			for j in p: #for each checker in that position
				if (p.index(j) >= 3):
					inputs[((i-1)*8)+p.index(j)] = ((len(p)-3)/2)
					break
				else:
					inputs[((i-1)*8)+p.index(j)] = 1
			#end for
		else: #if this is NOT your color, add +4 to the position, as an offset, that represents the area reserved for black
			for j in p: #for each checker in that position
				if (p.index(j) >= 3):
					inputs[((i-1)*8)+p.index(j)+4] = ((len(p)-3)/2)
					break
				else:
					inputs[((i-1)*8)+p.index(j)+4] = 1
			#end for
	if (color == WHITE): #append a input to determine if your color is white
		inputs[192] = 1
	if (color == BLACK):#this input determines if you're playing with black
		inputs[193] = 1
		
	enemyColor = otherColor(color)
	inputs[194] = (len(game.board.jailed(color))/2) #one half the amount of jailed pieces you have
	inputs[195] = (len(game.board.jailed(enemyColor))/2) #one half the amount of jailed pieces for the opponent
	inputs[196] = (len(game.board.homed(color)))
	inputs[197] = (len(game.board.homed(enemyColor)))
	#--- end of inputs ---
	#print(inputs)
	return inputs

class NNPlayer():
	"""
	Our own neural network player
	"""

	def __init__(I, color):
		I.color = color
		I.old_game = ""
		
	def interact(I, game):
		if (I.color == WHITE): #if this is white's turn
			#figures out the best move for white
			print("white's turn")
			high_score = 0
			best_moves = []
			for moves in game.all_choices():
				copyGame = copy.deepcopy(game)
				#copyGame.board = copyGame.board.copy()
				for m in moves:
					copyGame.move(*m)
				score = evaluateMove(copyGame, I.color)[0]
				#print("SCORE: {:5}	 PATH: {}".format(score, moves))
				if score > high_score:
					high_score = score
					best_moves = moves
					
			for move in best_moves: #and then perform the best move on the board
				print("MOVE:", move, high_score, I.color)
				game.draw()
				game.move(*move)
			
			#train the NN based on the old board, and this new evaluation
			if I.old_game != "" and best_moves != []: #if this isn't the first move
				if (game.board.finished()):
					trainNetwork(I.old_game,1,otherColor(I.color)) #game's over, one last training with desired_output = 1, because this side won
				else:				
					trainNetwork(I.old_game,high_score,I.color)
			
			I.old_game = copy.deepcopy(game) #saves the current board as the old one, for the next turn
		
		else: #if this is black's turn
			#find the worst move as black
			print ("black's turn")
			low_score = 1
			worst_moves = []
			for moves in game.all_choices():
				copyGame = copy.deepcopy(game)
				#copyGame.board = copyGame.board.copy()
				for m in moves:
					copyGame.move(*m)
				score = evaluateMove(copyGame, otherColor(I.color))[0]
				#print("SCORE: {:5}	 PATH: {}".format(score, moves))
				#print score, low_score
				if score < low_score:
					low_score = score
					worst_moves = moves
			
			for move in worst_moves: #and then perform the best move on a temp board
				print("MOVE:", move, low_score, I.color)
				game.draw()
				game.move(*move)
				
			#train the NN based on the old board, and this new evaluation, trains always as white
			if I.old_game != "" and worst_moves != []:
				if (game.board.finished()):
					trainNetwork(I.old_game,0,I.color) #game's over, one last training with desired_output = 0, because this side won
				else:
					trainNetwork(I.old_game,low_score,I.color) #1-low_score means the white's chance of winning, based on the black's chance of winning
			I.old_game = game

if __name__ == '__main__':
	neuralNetwork = bpnn.NN(198,50,1)
	#if we have a file with the latest weights for the network, load it into the NN object
	#neuralNetwork.loadWeightsFromFile("nn")			
	
	for i in range(1): #number of games you want to play
		game = Game()
		game.white = NNPlayer(WHITE)
		game.black = NNPlayer(BLACK)
		game.play()
		
		
	#after playing those games, save the weights in a file
	#neuralNetwork.writeWeightsToFile("nn")
