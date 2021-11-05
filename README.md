# QLearning - Reinforment Learning 
## Pacman Small-Grid
#### Note this project is performed on Python2.

![image](https://user-images.githubusercontent.com/60708644/124385152-805dc480-dccc-11eb-82a8-e4274788c06d.png)

mlLearningAgents.py contains a class QLearnAgent, and that class includes the following methods.
* init() This is the constructor for QLearnAgent. It is called by the game when the game starts up
(because the game starts up the learner).
* The version of init() in QLearnAgent allows you to pass parameters from the command
line. Some of these you know from the lectures:
  - alpha, the learning rate
  - gamma, the discount rate  
  - epsilon, the exploration rate
and you will use them in the implementation of your reinforcement learning algorithm. The
other:
  - numTraining allows you to run some games as training episodes and some as real games.
* All the constructor does is take these values from the command line, if you pass them, and
write them into sensibly named variables. If you don’t pass values from the command line,
they take the default values you see in the code. 
* These values work perfectly well, but if you
want to play with different values, then you can do that like this:
python pacman.py -p QLearnAgent -l smallGrid -a numTraining=2 -a alpha=0.2
* Note that you need to have no space between parameter and value alpha=0.2, and you need
a separate -a for each one.
  getAction() This function is called by the game every time that it wants Pacman to make a move
(which is every step of the game).
* This is the core of code that controls Pacman. It has access to information about the position
of Pacman, the position of ghosts, the location of food, and the score. The code shows how
to access all of these, and if you run the code (as above) all these values are printed out each
time getAction() is called.
* The only bit that maybe needs some explanation is the food. Food is represented by a grid of
letters, one for each square in the game grid. If the letter is F, there is no food in that square.
If the letter is T, then there is food in that square. (Walls are not represented).

##### Example code: python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
##### The above code will Train the Temporal Diffrence Learning algorithm for 2000 iterations, and will Test the Pacman game for 10 times. 
