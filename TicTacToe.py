import cv2
import numpy as np
import math
import mediapipe as mp
import copy
import time
import pygame

def CreateUI(BoardState, UIHeight, CellSize, X, O,HoverIndex, HoverProgress,HFrame,WFrame):
    UIBG = np.zeros((HFrame, WFrame, 3), dtype=np.uint8)
    BoardTopLeft = ((WFrame-UIHeight)//2,(HFrame-UIHeight)//2)
    for i in range(1, 3):
        cv2.line(UIBG, (BoardTopLeft[0]+i * CellSize, BoardTopLeft[1]), (BoardTopLeft[0]+i * CellSize, BoardTopLeft[1]+UIHeight), (255, 255, 255), 4)
        cv2.line(UIBG, (BoardTopLeft[0], BoardTopLeft[1]+i * CellSize), (BoardTopLeft[0]+UIHeight, BoardTopLeft[1]+i * CellSize), (255, 255, 255), 4)

    if HoverIndex != -1 and HoverProgress > 0:
        row, col = HoverIndex // 3, HoverIndex % 3
        cell_x, cell_y = BoardTopLeft[0] + col * CellSize, BoardTopLeft[1] + row * CellSize
        
        bar_height = int(CellSize * HoverProgress)
        
        cv2.rectangle(UIBG, (cell_x, cell_y + CellSize - bar_height), 
                      (cell_x + CellSize, cell_y + CellSize), 
                      (0, 150, 0), -1)

    for r in range(3):
        for c in range(3):
            center_x, center_y = BoardTopLeft[0] + c * CellSize + CellSize // 2, BoardTopLeft[1] + r * CellSize + CellSize // 2
            if BoardState[r][c] == X:
                cv2.line(UIBG, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (255, 0, 0), 6)
                cv2.line(UIBG, (center_x + 50, center_y - 50), (center_x - 50, center_y + 50), (255, 0, 0), 6)
            elif BoardState[r][c] == O:
                cv2.circle(UIBG, (center_x, center_y), 50, (0, 0, 255), 6)
    return UIBG

def Touch(frame, WFrame, HFrame, BoardTopLeft, CellSize, hand, mpdrawing, mphands):
    FingerOption = -1
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    MpResult = hand.process(RGB)

    IndexPos = (-20,-20)
    
    if MpResult.multi_hand_landmarks:
        HandLandmarks = MpResult.multi_hand_landmarks[0]
        IndexTip, ThumbTip = HandLandmarks.landmark[mphands.HandLandmark.INDEX_FINGER_TIP], HandLandmarks.landmark[mphands.HandLandmark.THUMB_TIP]
        IndexPos = (int(IndexTip.x * WFrame), int(IndexTip.y * HFrame))
        
        BoardX, BoardY = BoardTopLeft
        BoardSize = CellSize * 3
        if BoardX < IndexPos[0] < BoardX + BoardSize and BoardY < IndexPos[1] < BoardY + BoardSize:
            col = (IndexPos[0] - BoardX) // CellSize
            row = (IndexPos[1] - BoardY) // CellSize
            FingerOption = row * 3 + col
            
    return FingerOption, IndexPos

def InitialState(EMPTY): return [[EMPTY] * 3 for _ in range(3)]
def Player(board,X,O):
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count <= o_count else O
def actions(board,EMPTY): return {(r, c) for r in range(3) for c in range(3) if board[r][c] == EMPTY}
def result(board, action,EMPTY,X,O):
    if board[action[0]][action[1]] is not EMPTY: raise ValueError("Invalid")
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = Player(board,X,O)
    return new_board
def winner(board,EMPTY):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != EMPTY: return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != EMPTY: return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != EMPTY: return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY: return board[0][2]
    return None
def terminal(board,EMPTY): return winner(board,EMPTY) is not None or not any(EMPTY in row for row in board)
def utility(board,X,O,EMPTY):
    win = winner(board,EMPTY)
    if win == X: return 1
    elif win == O: return -1
    else: return 0
def min_value(board,EMPTY,X,O):
    if terminal(board,EMPTY): return utility(board,X,O,EMPTY)
    v = float("inf")
    for action in actions(board,EMPTY): v = min(v, max_value(result(board, action,EMPTY,X,O),EMPTY,X,O))
    return v
def max_value(board,EMPTY,X,O):
    if terminal(board,EMPTY): return utility(board,X,O,EMPTY)
    v = float("-inf")
    for action in actions(board,EMPTY): v = max(v, min_value(result(board, action,EMPTY,X,O),EMPTY,X,O))
    return v
def minimax(board,X,EMPTY,O):
    if terminal(board,EMPTY): return None
    current_player = Player(board,X,O)
    if current_player == X: return max(actions(board), key=lambda action: min_value(result(board, action)))
    else: return min(actions(board,EMPTY), key=lambda action: max_value(result(board, action,EMPTY,X,O),EMPTY,X,O))

def TicTacToeMain(cam):
    X = "X"
    O = "O"
    EMPTY = None
    UIHeight = 600
    CellSize = UIHeight // 3
    board = InitialState(EMPTY)
    user_player = X

    pygame.mixer.init()
    PopSound = pygame.mixer.Sound("./sounds/TicTacToe/PopSound.mp3")

    EndState = False
    EndDelay = 3.0
    EndTime = 0

    BotThinking, BotThinkingStart, BotDelay = False, 0, 1.5

    Alpha = 1
    mphands, mpdrawing = mp.solutions.hands, mp.solutions.drawing_utils
    hand = mphands.Hands(min_detection_confidence=0.7, max_num_hands=1)

    HoveredButton = -1
    HoverStart = 0
    HoverDuration = 2

    StartTime = time.time()
    StartDelay = 5.0

    while True:
        success, frame = cam.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        HFrame, WFrame, _ = frame.shape
        key = cv2.waitKey(1) & 0xFF
        
        if time.time() < StartTime+StartDelay:
            message = "Get Ready..."
            (TextWidth,TextHeight),_ = cv2.getTextSize(message,cv2.FONT_HERSHEY_SIMPLEX,1.5,3)
            TextX = (WFrame-TextWidth)//2
            TextY = (HFrame+TextHeight)//2

            cv2.putText(frame,message,(TextX,TextY),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)

            cv2.imshow("VideoFeed", frame)
            continue

        BoardTopLeft = ((WFrame - UIHeight) // 2, (HFrame - UIHeight) // 2)
        
        FingerOption, IndexPos = Touch(frame, WFrame, HFrame, BoardTopLeft, CellSize, hand, mpdrawing, mphands) # BoxDims is not needed here
        HoverProgress = 0

        GameOver = terminal(board,EMPTY)
        CurrentPlayer = Player(board,X,O)
        CurrentTime = time.time()

        if GameOver and not EndState:
            EndState = True
            EndTime = time.time()
        
        if not GameOver and CurrentPlayer == user_player and not BotThinking:
            if FingerOption != -1:
                if FingerOption == HoveredButton:
                    HoverElapsed = CurrentTime-HoverStart
                    HoverProgress = min(HoverElapsed/HoverDuration,1.0)

                    if HoverElapsed >= HoverDuration:
                        row, col = FingerOption // 3, FingerOption % 3
                        if board[row][col] == EMPTY:
                            board = result(board, (row, col),EMPTY,X,O)
                            LastPinch = CurrentTime
                            if not terminal(board,EMPTY):
                                BotThinking = True
                                BotThinkingStart = CurrentTime
                        PopSound.play()
                        HoveredButton = -1
                        HoverStart = 0
                else:
                    HoveredButton = FingerOption
                    HoverStart = CurrentTime
            else:
                HoveredButton = -1
                HoverStart = 0
        
        UIBG = CreateUI(board, UIHeight, CellSize, X, O, HoveredButton, HoverProgress, HFrame, WFrame)

        if BotThinking and (CurrentTime > BotThinkingStart + BotDelay):
            move = minimax(board,X,EMPTY,O)
            board = result(board, move,EMPTY,X,O)
            BotThinking = False
            PopSound.play()
        
        # --- Message Display ---
        message = ""
        if GameOver:
            win = winner(board,EMPTY)
            if win is None: message = "Game Over: Tie!"
            else: message = f"Game Over: {win} wins!"
        elif BotThinking: message = "Bot is thinking..."
        else: message = f"Your Turn ({user_player})"
        
        (TextWidth, TextHeight), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        TextX, TextY = (WFrame - TextWidth) // 2, TextHeight + 50
        cv2.putText(UIBG, message, (TextX, TextY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.circle(UIBG,IndexPos,25,(0,0,255),-1)

        cv2.imshow("VideoFeed", UIBG)
        if key == ord('q'): break

        if EndState and (CurrentTime > EndTime+EndDelay):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    TicTacToeMain()