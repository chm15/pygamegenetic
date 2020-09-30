"""
 Machine Learning Flappy Bird by Connor McLaughlan (10/23/19)
    -Learns using genetic algorithms.
"""




import math, pygame, time, sys, random, numpy as np
from pprint import pprint

pygame.init()
random.seed()
randTime = int(time.time() // 1)
np.random.seed(randTime)

metersToPixels = 50
globalXVelocity = 8

class Network:
    # Contains functions and structure for neural network.
    def __init__(self, dimensions, activations):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)  ex. [2, 3, 1],  len(ex.) = 3
        :param activations: (tpl/ list) Activations functions.

        """

        self.totalLayers = len(dimensions)
        self.loss = None
        self.learningRate = None

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2].
        self.w = {}
        self.b = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            self.w[i + 1] = np.random.rand(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]

    def feedForward(self, x):
        """
        Execute a forward feed through the network

        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer.
                 The numbering of the output is equivalent to the layer numbers.
        """

        # w(x) + b = z
        z = {}

         # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.totalLayers):
            # current layer = i
            # activation layer = i + 1
            z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])

        """
        a = {
            1: "inputs x",
            2: "activations of relu function in the hidden layer",
            3: "activations of the sigmoid function in the output layer"
        }

        z = {
            2: "z values of the hidden layer",
            3: "z values of the output layer"
        }
        """

        return z, a

    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        _, a = self.feedForward(x)
        return a[self.totalLayers]
    
    def randomize(self, rate):
        """
        Randomize weights and biases.
        """
        for i in range(self.totalLayers - 1):
            for j in range(len(self.w[i + 1])):
                for k in range(len(self.w[i + 1][j])):
                    chanceOfMutation = random.randint(0, 5)
                    if chanceOfMutation == 0:
                        self.w[i + 1][j][k] += (random.randint(0, 600) - 300 )/ (1000 + (100* rate) )
        for i in range(len(self.b)):
            for j in range(len(self.b[i + 1])):
                chanceOfMutation = random.randint(0, 5)
                if chanceOfMutation == 0:
                    self.b[i + 1][j] += (random.randint(0, 600) - 300)/ (1000 + (100 * rate) )



class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

class Player:
    # Contains physics calculations and functions for drawing player to screen. Has neural network object included.
    def __init__(self, playerId, startingY):
        self.startingY = startingY
        self.setStartingValues()
        self.playerId = playerId
        #self.playerColor = (180, 100, 100)
        self.playerColor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.network = Network((2, 3, 1), (Sigmoid, Sigmoid, Sigmoid))

    def setStartingValues(self):
        self.startTime = time.time()
        self.flapTime = time.time()
        self.dead = False
        self.size = 50
        self.xWallDisplacement = 40 # Pixels
        self.Vx = globalXVelocity
        self.Vyi = -12
        self.Vy = self.Vyi
        self.x = 0
        self.y = 0
        self.a = 40
        self.xMeters = 2
        self.yMeters = 6
        self.dy = 0
        self.yInitial = self.startingY
        self.verticalDifferenceOfPipe = 0
        self.distanceToPipe = 0
        self.score = 0

    


    def flap(self, currentTime):
        timeDifference = currentTime - self.flapTime
        if timeDifference > 0.2:
            self.flapTime = currentTime
            self.yInitial = self.yMeters

    def runPhysics(self, currentTime):
        if not self.dead:
            newTime = currentTime
            dt = newTime - self.startTime

            # Update x coords.
            self.xMeters = dt * self.Vx

            # Update y coords.
            timeSinceFlap = newTime - self.flapTime
            self.dy = (self.Vyi * timeSinceFlap) + (0.5 * self.a * (timeSinceFlap**2))
            self.yMeters = self.yInitial + self.dy

            # Update player's vertical velocity. Negative is downwards.
            self.Vy = -(self.Vyi + (self.a * (time.time() - self.flapTime)))

            # Map the measurements to pixels.
            self.x = self.distanceMap(self.xMeters) + self.xWallDisplacement
            self.y = self.distanceMap(self.yMeters)
    
    def distanceMap(self, meters):
        return meters * metersToPixels

    def draw(self, screen, scrollDistance):
        pygame.draw.rect(screen, self.playerColor, (self.x - scrollDistance, self.y, self.size, self.size))

    def runNetwork(self, currentTime):
        velocityNormalized = self.Vy
        prediction = self.network.predict([self.verticalDifferenceOfPipe, velocityNormalized])
        if prediction > 0.5:
            self.flap(currentTime)




class Game:
    # This class is used to run the game. Run the game with a list of players
    
    framerate = 1000
    screenSize = screenWidth, screenHeight = 900, 700
    screen = pygame.display.set_mode(screenSize)

    startingHeight = 20
    scrollDistance = 0
    scrollVelocity = globalXVelocity # m/s, must be mapped to pixels.
    gravity = 9.81
    startTime = 0

    players = []
    gameOver = False

    pipeGap = 170
    pipeWidth = 100
    distanceBetweenPipes = 600

    pipes = [[screenWidth, (screenHeight + pipeGap)/2]]
    pipeColor = (100, 200, 100)
    pipeMinMax = 30

    textFont = pygame.font.SysFont('Arial', 100)

    backgroundColor = (30, 40, 50)

    def startGame(self):
        self.gameOver = False
        self.scrollDistance = 0
        self.pipes = [[self.screenWidth, (self.screenHeight + self.pipeGap)/2]]


        for player in self.players:
            player.startTime = time.time()
        self.startTime = time.time()
        while not self.gameOver:

            currentTime = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                if event.type == pygame.KEYDOWN:
                    self.players[0].flap(currentTime)


            # Run feed forward neural network for each player.
            for player in self.players:
                if not player.dead:
                    pipex = None
                    pipey = None
                    for pipe in self.pipes:
                        endOfPipe = pipe[0] + self.pipeWidth
                        if endOfPipe > player.x:
                            pipex = endOfPipe
                            pipey = pipe[1]
                            break

                    player.distanceToPipe = pipex - player.x
                    vertDifference = player.y - pipey
                    player.verticalDifferenceOfPipe = vertDifference
                    
                    player.runNetwork(currentTime)

            # Run physics for all players and map.
            self.scrollDistance = self.distanceMap(self.scrollVelocity * (currentTime - self.startTime))
            
            for player in self.players:
                player.runPhysics(currentTime)

            
            # Check for colisions.
            for pipe in self.pipes:
                for player in self.players:
                    pipexScreen = pipe[0] - self.scrollDistance
                    playerxScreen = player.x - self.scrollDistance
                    if pipexScreen < playerxScreen + player.size and pipexScreen + self.pipeWidth > playerxScreen:
                        if player.y + player.size > pipe[1] or player.y < pipe[1] - self.pipeGap:
                            player.dead = True
                    if player.y + player.size > self.screenHeight:
                        player.dead = True



            # Generate new pipes.
            lastPipe = self.pipes[-1][0] - self.scrollDistance
            if lastPipe < self.screenWidth:
                randomHeight = random.randint(self.pipeMinMax + self.pipeGap, self.screenHeight - self.pipeMinMax)
                newPipex = self.pipes[-1][0] + self.distanceBetweenPipes
                newPipey = randomHeight
                self.pipes.append([newPipex, newPipey])

            # Draw all players and objects.
            self.screen.fill(self.backgroundColor)
            for player in self.players:
                player.draw(self.screen, self.scrollDistance)

            for pipe in self.pipes:
                pipeScreenCoords = pipe[0] - self.scrollDistance
                if pipeScreenCoords < self.screenWidth and pipeScreenCoords + self.pipeWidth > 0:
                    pygame.draw.rect(self.screen, self.pipeColor, 
                            (pipe[0] - self.scrollDistance, pipe[1], self.pipeWidth, self.screenHeight - pipe[1]))
                    pygame.draw.rect(self.screen, self.pipeColor,
                            (pipe[0] - self.scrollDistance, 0, self.pipeWidth, pipe[1] - self.pipeGap))


            # Draw score to screen.



            score = 0

            # Test for game over condition.
            self.gameOver = True
            for player in self.players:
                if player.dead == False:
                    self.gameOver = False
                    for pipe in self.pipes:
                        if pipe[0] < player.x:
                            score += 1
                    break

            currentScore = self.textFont.render(str(score), False, (250, 250, 250))
            self.screen.blit(currentScore, ((self.screenWidth / 2) - 20, 30))

            if score > 10:
                self.gameOver = True
           
            # Update Screen.
            pygame.display.flip()

            # Pause loop for fraction of a second.
            time.sleep(1/self.framerate)

        # Game over processes.
        #return 0
        for player in self.players:
            score = 0
            for pipe in self.pipes:
                if pipe[0]  < player.x:
                    score += 1
            #player.score = score
            player.score = player.x //50
    
    def distanceMap(self, meters):
        return meters * metersToPixels



# ========================================================================================


def runGameWithNetworks():

    game = Game()

    totalPlayers = 10
    totalIterations = 100

    # Append each player with starting y positions to the game.
    for x in range(totalPlayers):
        startingY = ((game.screenHeight - 100) / (metersToPixels *  totalPlayers)) * x
        playerObject = Player(x, startingY)
        game.players.append(playerObject)

    # Run the game for a total of x iterations, then sort 
    highScore = 0
    highScoreId = None
    for x in range(totalIterations):
        print("Gen: ", x + 1)
        game.startGame()
        game.players = sortPlayers(game.players)

        #for player in game.players:
        #    if player.score > highScore:
        #        highScore = player.score
        #        highScoreId = player.playerId
        #    if highScoreId != None and player.playerId == highScoreId:
        #        print(player.network.w)
        scoresSorted = []
        for player in game.players:
            scoresSorted.append(player.score)
        #print(scoresSorted)

        for i in range(len(game.players)):
            player = game.players[i]
            playerToPassOnGenes = i % 5
            if i > 4 and game.players[playerToPassOnGenes].score > player.score:
                playerToCopy = game.players[playerToPassOnGenes]
                weight = playerToCopy.network.w
                bias = playerToCopy.network.b
                for j in range(len(weight)):
                    game.players[i].network.w[j + 1] = weight[j + 1].copy()
                for j in range(len(bias)):
                    game.players[i].network.b[j + 1] = bias[j + 1].copy()
            if i > 4:
                game.players[i].network.randomize(x)

        for player in game.players:
            player.setStartingValues()

        print("\n")

def sortPlayers(players):
    sortedPlayers = []
    for player in players:
        score = player.score
        if len(sortedPlayers) == 0:
            sortedPlayers.append(player)
            continue
        else:
            for i in range(len(sortedPlayers)):
                otherScore = sortedPlayers[i].score
                if score > otherScore:
                    sortedPlayers.insert(i, player)
                    break
                elif score < otherScore and i == (len(sortedPlayers) - 1):
                    sortedPlayers.append(player)
                    break
                elif score == otherScore:
                    sortedPlayers.insert(i, player)
                    break
    return sortedPlayers


# Run the game.
runGameWithNetworks()

