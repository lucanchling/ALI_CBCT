import numpy as np

LABELS = {
    "u" : ['PNS','ANS','A','UR6apex','UR3apex','U1apex','UL3apex','UL6apex','UR6d','UR6m','UR3tip','UItip','UL3tip','UL6m','UL6d'],
    # "l" : ['RCo','RGo','LR6apex','LR7apex','L1apex','Me','Gn','Pog','B','LL6apex','LL7apex','LGo','LCo','LR6d','LR6m','LItip','LL6m','LL6d'],
    "l" : ['RCo','RGo','LR6apex','L1apex','Me','Gn','Pog','B','LL6apex','LGo','LCo','LR6d','LR6m','LItip','LL6m','LL6d'],
    "cb" :['Ba','S','N']
}

MOVEMENT_MATRIX_6 = np.array([
    [1,0,0],  # MoveUp
    [-1,0,0], # MoveDown
    [0,1,0],  # MoveBack
    [0,-1,0], # MoveFront
    [0,0,1],  # MoveLeft
    [0,0,-1], # MoveRight
])

MOVEMENT_MATRIX_26 = np.array([
    [1,0,0],   # MoveUp
    [1,1,0],   # MoveUpBack
    [1,-1,0],  # MoveUpFront
    [1,0,1],   # MoveUpLeft
    [1,0,-1],  # MoveUpRight
    [1,1,1],   # MoveUpBackLeft
    [1,1,-1],  # MoveUpBackRight
    [1,-1,1],  # MoveUpFrontLeft
    [1,-1,-1], # MoveUpFrontRight
    [-1,0,0],  # MoveDown
    [-1,1,0],  # MoveDownBack
    [-1,-1,0], # MoveDownFront
    [-1,0,1],  # MoveDownLeft
    [-1,0,-1], # MoveDownRight
    [-1,1,1],  # MoveDownBackLeft
    [-1,1,-1], # MoveDownBackRight
    [-1,-1,1], # MoveDownFrontLeft
    [-1,-1,-1],# MoveDownFrontRight
    [0,1,0],   # MoveBack
    [0,1,1],   # MoveBackLeft
    [0,1,-1],  # MoveBackRight
    [0,-1,0],  # MoveFront
    [0,-1,1],  # MoveFrontLeft
    [0,-1,-1], # MoveFrontRight
    [0,0,1],   # MoveLeft
    [0,0,-1]   # MoveRight
])

MOVEMENT_ID_6 = ["Up", "Down", "Back", "Front", "Left", "Right"]
MOVEMENT_ID_26 = [
    "Up",
    "UpBack",
    "UpFront",
    "UpLeft",
    "UpRight",
    "UpBackLeft",
    "UpBackRight",
    "UpFrontLeft",
    "UpFrontRight",
    "Down",
    "DownBack",
    "DownFront",
    "DownLeft",
    "DownRight",
    "DownBackLeft",
    "DownBackRight",
    "DownFrontLeft",
    "DownFrontRight",
    "Back",
    "BackLeft",
    "BackRight",
    "Front",
    "FrontLeft",
    "FrontRight",
    "Left",
    "Right"
]

def GetTargetOutputFromAction(action):
    target = np.zeros((1,len(MOVEMENT_MATRIX)))[0]
    target[action] = 1
    return target