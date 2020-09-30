import math, pygame, time, sys, random, numpy as np
from pprint import pprint

pygame.init()
random.seed()
randTime = int(time.time() // 1)
np.random.seed(randTime)

metersToPixels = 20 #20

class Game:
    def __init__(self, players):
        """ 
        Contains the main game. Call startGame() to begin.
        :players = [Player()]

        """
        self.maxTime = 3
        self.players = players
        self.loopPause = 10000000 # 1/x
        self.backgroundColor = (30, 40, 50)
        self.screenSize = self.screenWidth, self.screenHeight = 900, 700
        self.screen = pygame.display.set_mode(self.screenSize)
        self.groundHeight = 10  # Pixels
        self.groundPosition = self.screenHeight - self.groundHeight
        self.groundColor = (70, 80, 110)
        self.furthestxStartPoint = players[0].startingX * metersToPixels
        self.furthestx = self.furthestxStartPoint
        self.textFont = pygame.font.SysFont('Arial', 50)

    def startGame(self):
        startTime = time.time()
        self.initializePlayers(startTime)
        gameOver = False
        currentRelease = 0
        while not gameOver:
            currentTime = time.time()
            timeSinceStart = currentTime - startTime

            self.screen.fill(self.backgroundColor)

            # Check for pygame events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.players[currentRelease].releaseFromRope()
                        if currentRelease < len(self.players) - 1: currentRelease += 1

            # Iterate through all players and run physics, neural network, and then draw to screen.
            gameOver = True
            for player in self.players:

                # Check for gameOver Conditions.
                if (not player.dead and not player.isSpinning) or (timeSinceStart < self.maxTime and not player.dead):
                    gameOver = False

                # Run through player processes.
                if not player.dead:
                    # Physics for all objects in scene.
                    player.runPhysics(currentTime)
                    if player.x > self.furthestx:
                        self.furthestx = player.x

                    # Check for collisions.
                    if player.y + player.mapToScreen(player.playerRadius) > self.groundPosition:
                        player.dead = True

                    # Network prediction for each player.
                    if timeSinceStart < self.maxTime:
                        if player.network.predict(player.currentAngle%(2*math.pi)) > 0.5:
                            player.releaseFromRope()

                # Draw all objects to screen.
                player.draw(self.screen)

            # Draw static objects to screen.
            pygame.draw.rect(self.screen, self.groundColor,(0, self.screenHeight - self.groundHeight, self.screenWidth, self.screenHeight))
            pygame.draw.rect(self.screen, (220, 80, 80),(self.furthestx, 0, 3, self.screenHeight))
            currentScore = self.textFont.render(str(round((self.furthestx - self.furthestxStartPoint) / metersToPixels, 1)) + " m", False, (250, 250, 250))
            self.screen.blit(currentScore, (self.furthestx + 5, self.screenHeight - self.groundHeight - 40))


            rotationx = (players[0].startingX - players[0].radiusOfOrbit) * metersToPixels
            rotationy = players[0].startingY * metersToPixels
            pygame.draw.circle(self.screen, (110, 120, 135), (rotationx, rotationy), 5)

            # Update Screen.
            pygame.display.flip()

            # Check for gameOver conditions.
            if gameOver:
                time.sleep(0.25)
            
            # Pause loop for set time.
            time.sleep(1/self.loopPause)

        # ================= End of game loop. =====================
        return

    def initializePlayers(self, startTime):
        """ 
        :startTime = The starting time to synchronize all players to.
        """
        for player in self.players:
            player.startTime = startTime
            player.resetValues()


class Player:
    def __init__(self, playerId):
        """
        Player with all physics calculations built in.
        :startingCoords = (x, y) of starting coordinates in meters.
        The startingX and startingY values will be stored for use when the resetValues() function is called.
        """
        startingCoords = (7, 30)
        self.playerId = playerId
        self.startTime = None
        self.startingX = startingCoords[0]  # in meters.
        self.startingY = startingCoords[1] # in meters.
        self.velocity = 17 # m/s, must be mapped to pixels.
        self.radiusOfOrbit = 3 # in meters.
        self.rotationOrigin = (self.startingX - self.radiusOfOrbit, self.startingY)
        self.playerRadius = 0.5 # Radius of the player.
        self.acceleration = 9.81
        self.color = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))
        self.network = Network((1,3,1), (Sigmoid, Sigmoid))

        self.resetValues()

    def releaseFromRope(self):
        if self.isSpinning == True:
            self.isSpinning = False
            self.coordsAtRelease = (self.x, self.y)
            self.timeOfRelease = self.currentTime

            self.angleOfRelease = self.currentAngle + (math.pi / 2)
            Vx = self.velocity * math.cos(self.angleOfRelease)
            Vy = self.velocity * math.sin(self.angleOfRelease)
            self.velocitiesAtRelease = (Vx, Vy)


    def resetValues(self):
        self.x = self.startingX
        self.y = self.startingY
        self.isSpinning = True
        self.dead = False
        self.timeOfRelease = None
        self.coordsAtRelease = None
        self.angleOfRelease = None
        self.currentAngle = 0
        self.velocitiesAtRelease = None  # The x velocity parallel to the ground. Set upon release.
        self.score = 0

    def runPhysics(self, currentTime):
        # Update the player's x and y values, update the player's local time..
        self.currentTime = currentTime
        if not self.dead:
            if self.isSpinning:
                dx, dy = self.originOfOrbit(self.rotationOrigin, self.velocity, self.radiusOfOrbit)
                self.x = self.mapToScreen(self.startingX + dx)
                self.y = self.mapToScreen(self.startingY + dy)
            else:
                dtSinceRelease = currentTime - self.timeOfRelease
                self.x = (self.mapToScreen(dtSinceRelease * self.velocitiesAtRelease[0])) + self.coordsAtRelease[0]
                self.y = self.mapToScreen((self.velocitiesAtRelease[1] * dtSinceRelease) + (0.5 * self.acceleration * (dtSinceRelease**2))) + self.coordsAtRelease[1]
        self.getScore() # Updates the player's score

    def originOfOrbit(self, coords, velocity, radius):
        """
        Function returns (x, y) in units of meters of a point at (currentTime - startTime).
        :coords = (x, y) Coordinates to orbit around. 
        :velocity = Centripetal velocity of the object.
        :radius of circle.
        """
        dt = self.currentTime - self.startTime
        period = (2 * math.pi * radius) / velocity
        angle = (2 * math.pi * dt) / period
        self.currentAngle = angle

        x = radius * math.cos(angle) - radius #coordinate system translated left so that at dt = 0, x = 0.
        y = radius * math.sin(angle)

        return (x, y)

    def mapToScreen(self, meters):
        return int(meters * metersToPixels)
    
    def draw(self, screen):
        center = (int(self.x), int(self.y))
        radiusInPixels = self.mapToScreen(self.playerRadius)
        pygame.draw.circle(screen, self.color, center, radiusInPixels)

    def getScore(self):
        score = self.x - self.mapToScreen(self.startingX)
        if self.isSpinning:
            self.score = 0
        elif score < -5:
            self.score = 5
        else:
            self.score = self.x - self.mapToScreen(self.startingX)

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
                    chanceOfMutation = random.randint(0, 5*rate)
                    if chanceOfMutation == 0:
                        self.w[i + 1][j][k] += (random.randint(0, 600) - 300 )/ (1000  )
                        print((random.randint(0, 600) - 300 )/ (2000 ))
        for i in range(len(self.b)):
            for j in range(len(self.b[i + 1])):
                chanceOfMutation = random.randint(0, 5 * rate)
                if chanceOfMutation == 0:
                    self.b[i + 1][j] += (random.randint(0, 600) - 300)/ (1000  )

class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

def sortPlayersByScore(players):
    sortedPlayers = []
    for player in players:
        if len(sortedPlayers) == 0:
            sortedPlayers.append(player)
            continue
        for j in range(len(sortedPlayers)):
            score = player.score
            scoreSorted = sortedPlayers[j].score
            if score > scoreSorted:
                sortedPlayers.insert(j, player)
                break
            elif score == scoreSorted:
                sortedPlayers.insert(j, player)
                break
            elif j == len(sortedPlayers) - 1:
                sortedPlayers.append(player)
                break
            else:
                continue
    return sortedPlayers
                
def geneticAlgorithm(unsortedPlayers, generation):
    players = sortPlayersByScore(unsortedPlayers) 
    remainder = 10
    for i in range(remainder, len(players)):
        modulus = i%remainder
        playerToCopy = players[modulus]
        weight = playerToCopy.network.w
        bias = playerToCopy.network.b
        for j in range(len(weight)):
            players[i].network.w[j + 1] = weight[j + 1].copy()
        for j in range(len(bias)):
            players[i].network.b[j + 1] = bias[j + 1].copy()
        if i > len(players) // 4:
            players[i].network.randomize(generation)
    return players



totalPlayers = 100
totalGenerations = 100
players = []
for i in range(totalPlayers):
    players.append(Player(i))
game = Game(players)
for generation in range(totalGenerations):
    game.startGame()
    game.players = geneticAlgorithm(game.players, generation) 
    for player in players:
        player.resetValues()
